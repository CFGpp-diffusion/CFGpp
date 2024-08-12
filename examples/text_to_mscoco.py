import argparse
from pathlib import Path
import os

from munch import munchify
from torchvision.utils import save_image

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed


def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/mscoco")
    parser.add_argument('--prompt_dir', type=Path, default=Path('examples/assets/coco_v2.txt'))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--method", type=str, default='ddim')
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl", "sdxl_lightning"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    create_workdir(args.workdir)

    # load prompt
    text_list = []
    with open(args.prompt_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                text_list.append(stripped_line)
    text_list = text_list[:10000] # Test for 10k MS-COCO validation


    solver_config = munchify({'num_sampling': args.NFE })
    callback = ComposeCallback(workdir=args.workdir,
                               frequency=1,
                               callbacks=["draw_noisy", 'draw_tweedie'])
    # callback = None


    if args.model == "sdxl" or args.model == "sdxl_lightning":
        solver = get_solver_sdxl(args.method,
                                 solver_config=solver_config,
                                 device=args.device)

        for i, text in enumerate(text_list):
            print(f'Processing {i+1}/{len(text_list)}: {text}')

            result = solver.sample(prompt1=[args.null_prompt, text],
                                    prompt2=[args.null_prompt, text],
                                    cfg_guidance=args.cfg_guidance,
                                    target_size=(1024, 1024),
                                    callback_fn=callback)
            save_image(result, args.workdir.joinpath(f'{str(i).zfill(5)}.png'), normalize=True)

if __name__ == "__main__":
    main()