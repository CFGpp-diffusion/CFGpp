import argparse
from pathlib import Path
import logging
from typing import Optional
from functools import partial

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths


def prepare_logger(log_path: str):
    logger = logging.getLogger('Metric') 
    logger.setLevel(logging.INFO)
    
    stream = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s-%(levelname)s >> %(message)s')
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    
    filehandle = logging.FileHandler(log_path)
    logger.addHandler(filehandle)

    return logger


class ImagePathDataset(Dataset):
    def __init__(self, files, transforms=None) -> None:
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        path = self.files[index]
        img = Image.open(path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        return img
        
        
class TwoImagePathDataset(Dataset):
    def __init__(self, files1, files2, transforms=None) -> None:
        self.files1 = files1
        self.files2 = files2
        self.transforms = transforms

        assert len(self.files1) == len(self.files2), \
            f'Two file lists should have the same number of files. \
                Got {len(self.files1)} and {len(self.files2)}'

    def __len__(self):
        return len(self.files1)
    
    def __getitem__(self, index):
        path1 = self.files1[index]
        path2 = self.files2[index]

        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        return img1, img2
        


class Metric():
    def __init__(self, input_dir, label_dir, logger, device):
        self.input_dir = Path(input_dir)
        self.label_dir = Path(label_dir)

        self.logger = logger
        self.device = device

    def retrieve_img_paths(self, directory: Path):
        return sorted(list(directory.glob('*.png')))
    
    def compute(self):
        self.logger.info(f"Start to calculate metric {self}.")
        input_paths = self.retrieve_img_paths(self.input_dir)
        label_paths = self.retrieve_img_paths(self.label_dir)
        dataset = TwoImagePathDataset(files1=input_paths, 
                                      files2=label_paths,
                                      transforms=TF.ToTensor())
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False,
                                num_workers=4)
        
        result = [] 
        with torch.no_grad():
            for in_, label_ in tqdm(dataloader):
                in_ = self.preprocessing(in_)
                label_ = self.preprocessing(label_)

                if isinstance(in_, torch.Tensor):
                    in_ = in_.to(self.device)
                if isinstance(label_, torch.Tensor):
                    label_ = label_.to(self.device)
                
                value = self.metric_fn(label_, in_)
                if isinstance(value, np.float64): 
                    result.append(torch.tensor([value]))
                else:
                    result.append(value)
         
        mean = torch.stack(result).mean().detach().item()
        std = torch.stack(result).std().detach().item()

        self.logger.info(f"Result: mean={mean}  std={std}")
        
        return mean, std
        
    def preprocessing(self, img):
        return img

class MSE(Metric):
    def __init__(self, input_dir: str, label_dir: str, logger, device):
        super().__init__(input_dir, label_dir, logger, device)
        self.metric_fn = mse()

    def __str__(self) -> str:
        return 'MSE'

    def preprocessing(self, img):
        img = img.detach().numpy()
        return img
    

class LPIPS(Metric):
    def __init__(self, input_dir: str, label_dir: str, logger, device, net:Optional[str]='vgg'):
        super().__init__(input_dir, label_dir, logger, device)
        self.net = net
        self.metric_fn = lpips.LPIPS(net=net).to(self.device)
    
    def __str__(self):
        return 'LPIPS' 


class PSNR(Metric):
    def __init__(self, input_dir: str, label_dir: str, logger, device):
        super().__init__(input_dir, label_dir, logger, device)
        self.metric_fn = partial(psnr, data_range=255.0)
    
    def __str__(self) -> str:
        return 'PSNR'
        
    def preprocessing(self, img):
        img = img.detach().numpy()
        return img * 255


class FID(Metric):
    def __init__(self, input_dir: str, label_dir: str, logger, device):
        super().__init__(input_dir, label_dir, logger, device)
        self.metric_fn = calculate_fid_given_paths

    def __str__(self) -> str:
        return 'FID'

    def compute(self):
        self.logger.info(f"Start to calculate metric {self}.")
        value = self.metric_fn([str(self.input_dir), str(self.label_dir)],
                               batch_size=1,
                               device=self.device,
                               dims=2048)

        self.logger.info(f"Result: {value}")

class MNC(Metric):
    def __init__(self, input_dir: str, label_dir: str, logger, device):
        '''
        Maximum of normalized convolution by Z.Hu and M.-H Yang (ECCV 2012)
        '''
        super().__init__(input_dir, label_dir, logger, device)
        self.metric_fn = self.calculate_mnc

    def __str__(self) -> str:
        return 'MNC'

    def calcualte_mnc(self, estimated_kernel, true_kernel):
        assert estimated_kernel.shape[1] == 1, \
                f'Channel dimension of kernel should be 1, but got {estimated_kernel.shape[1]}.'
        assert true_kernel.shape[1] == 1, \
                f'Channel dimension of kernel should be 1, but got {true_kernel.shape[1]}.'

        value = F.conv2d(estimated_kernel, true_kernel, padding='same')
        value /= (torch.linalg.norm(estimated_kernel[0,0], ord=2) * torch.linalg.norm(true_kernel[0,0], ord=2))
        return value[0,0].max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--label_dir', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--log_path', type=str, default='./result.log')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    logger = prepare_logger(args.log_path)
    
    metrics = [FID(args.input_dir, args.label_dir, logger, device),
               LPIPS(args.input_dir, args.label_dir, logger, device), 
               PSNR(args.input_dir, args.label_dir, logger, device)]
    
    logger.info(f'============= Metric Calculation for {args.exp_name} =============')
    for metric in metrics:
        metric.compute() 
    

if __name__ == '__main__':
    main()
