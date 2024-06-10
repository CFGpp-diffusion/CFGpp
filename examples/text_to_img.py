import argparse
from pathlib import Path

from munch import munchify
from torchvision.utils import save_image

from solver.latent_diffusion import get_solver
from solver.latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed


def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/t2i")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--method", type=str, default='ddim')
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    create_workdir(args.workdir)

    solver_config = munchify({'num_sampling': args.NFE })
    callback = ComposeCallback(workdir=args.workdir,
                               frequency=1,
                               callbacks=["draw_sds_loss", 'draw_tweedie'])


    if args.model == "sdxl":
        solver = get_solver_sdxl(args.method,
                                 solver_config=solver_config,
                                 device=args.device)
        for i in range(13):
            result = solver.sample(prompt1=[args.null_prompt, args.prompt],
                                prompt2=[args.null_prompt, args.prompt],
                                cfg_guidance=args.cfg_guidance,
                                target_size=(1024, 1024),
                                callback_fn=callback)
            save_image(result, args.workdir.joinpath(f'result/generated.png'), normalize=True)
            import ipdb; ipdb.set_trace()
    else:
        solver = get_solver(args.method,
                            solver_config=solver_config,
                            device=args.device)
        result = solver.sample(prompt=[args.null_prompt, args.prompt],
                               cfg_guidance=args.cfg_guidance,
                               callback_fn=callback)

    
    save_image(result, args.workdir.joinpath(f'result/generated.png'), normalize=True)

if __name__ == "__main__":
    main()
