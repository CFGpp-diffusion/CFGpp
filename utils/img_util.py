from typing import Union, Optional
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from torchvision.utils import save_image
import numpy as np

def draw_img(img: Union[torch.Tensor, np.ndarray],
            save_path:Optional[str]='test.png',
            nrow:Optional[int]=8,
            normalize:Optional[bool]=True):
    if isinstance(img, np.ndarray):
        img = torch.Tensor(img)

    save_image(img, fp=save_path, nrow=nrow, normalize=normalize)

def normalize(img: Union[torch.Tensor, np.ndarray]) \
                        -> Union[torch.Tensor, np.ndarray]:
    
    return (img - img.min())/(img.max()-img.min())
     
def to_np(img: torch.Tensor,
          mode: Optional[str]='NCHW') -> np.ndarray:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        img = img.permute(0,2,3,1) 

    return img.detach().cpu().numpy()

def fft2d(img: torch.Tensor,
          mode: Optional[str]='NCHW') -> torch.Tensor:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        return fftshift(fft2(img))
    elif mode == 'NHWC':
        img = img.permute(0,3,1,2)
        return fftshift(fft2(img))
    else:
        raise NameError    
    

def ifft2d(img: torch.Tensor,
           mode: Optional[str]='NCHW') -> torch.Tensor:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        return ifft2(ifftshift(img))
    elif mode == 'NHWC':
        img = ifft2(ifftshift(img))
        return img.permute(0,2,3,1)
    else:
        raise NameError    

