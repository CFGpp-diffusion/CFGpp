from pathlib import Path

import torch
from torchvision.utils import save_image

__CALLBACK__ = {}

def register_callback(name):
    def wrapper(cls):
        if __CALLBACK__.get(name) is not None:
            raise NameError(f"Callback {name} is already registered")
        __CALLBACK__[name] = cls
        return cls
    return wrapper

def get_callback(name, **kwargs):
    if __CALLBACK__.get(name) is None:
        raise NameError(f"Callback {name} is not registered")
    return __CALLBACK__[name](**kwargs)

# Callback functions that are often used during diffusion process
class DiffusionCallback:
    def __init__(self, frequency: int, workdir: Path):
        """
        If frequency = 5, then the callback function will be called every 5 steps
        """
        assert frequency > 0, "Frequency must be a positive float"
        self.frequency = frequency
        self.workdir = workdir

    def __call__(self, step, t, callback_kwargs):
        if (step+1) % self.frequency == 0 or step == 0:
            return self.callback(step, t, callback_kwargs)
        return callback_kwargs
    
    def callback(self, step, t, callback_kwargs):
        raise NotImplementedError

@register_callback("draw_tweedie")
class DrawTweedieCallback(DiffusionCallback):
    def __init__(self, frequency: int, workdir: Path):
        super().__init__(frequency, workdir)
        workdir.joinpath("record/tweedie").mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def callback(self, step, t, callback_kwargs):
        z0t = callback_kwargs["z0t"]
        x0t = callback_kwargs["decode"](z0t)
        x0t = (x0t / 2 + 0.5).clamp(0, 1).cpu()
        save_image(x0t, self.workdir.joinpath(f"record/tweedie/x0_{int(t)}.png"))
        return callback_kwargs

@register_callback("draw_noisy")
class DrawNoisyCallback(DiffusionCallback):
    def __init__(self, frequency: int, workdir: Path):
        super().__init__(frequency, workdir)
        workdir.joinpath("record/noisy").mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def callback(self, step, t, callback_kwargs):
        z0t = callback_kwargs["zt"]
        x0t = callback_kwargs["decode"](z0t)
        x0t = (x0t / 2 + 0.5).clamp(0, 1).cpu()
        save_image(x0t, self.workdir.joinpath(f"record/noisy/xt_{int(t)}.png"))
        return callback_kwargs

class ComposeCallback(DiffusionCallback):
    def __init__(self, workdir, callbacks: list[str], frequency:int=5):
        super().__init__(frequency, workdir)
        self.callbacks = [get_callback(name, workdir=workdir, frequency=frequency) for name in callbacks]

    def __call__(self, step, t, callback_kwargs):
        for callback in self.callbacks:
            callback_kwargs = callback(step, t, callback_kwargs)
        return callback_kwargs