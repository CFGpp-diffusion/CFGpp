"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

####### Factory #######
__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

########################
# Helper functions
# taken from comfyui
########################

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

########################
# Base classes
########################

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        """
        The base class of LDM-based solvers for VP sampling.
        We load pre-trained VAE, text-encoder, and U-Net models from diffusers.
        Also, compute pre-defined coefficients.

        args:
            solver_config (Dict): solver configurations (e.g. NFE)
            model_key (str): model key for loading pre-trained models
            device (torch.device): device
            **kwargs: additional arguments
        """
        # pre-traiend model loading
        self.device = device
        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        # load scheduler
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        # time discretization  
        total_timesteps = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """
        The method that distinguishes each solver.
        """
        raise NotImplementedError("Solver must implement sample() method.")

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def get_text_embed(self, null_prompt: str, prompt: str):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt",)
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                    truncation=True)
        text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return null_text_embed, text_embed

    def encode(self, x: torch.Tensor):
        """
        Encode image to latent features.
        args:
            x (torch.Tensor): image
        """
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, zt: torch.Tensor):
        """
        Decode latent features to image.
        args:
            zt (torch.Tensor): latent
        """
        zt = 1/0.18215 * zt
        img = self.vae.decode(zt).sample.float()
        return img

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor):
        """
        compuate epsilon_theta for null and condition
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
        """
        t_in = t.unsqueeze(0)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float=1.0):

        # initialize z_0
        zt = z0.clone().to(self.device)

        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt

    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          latent_size: tuple=(1, 4, 64, 64),
                          **kwargs):
        """
        Initialize latent features.
        Simply, sample from Gaussian distribution or do inversion.
        args:
            method (str): initialization method
            src_img (torch.Tensor): source image
            **kwargs: additional arguments
        """
        if method == 'ddim':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               cfg_guidance=kwargs.get('cfg_guidance', 0.0))
        elif method == 'npi':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               cfg_guidance=1.0)
        elif method == 'random':
            z = torch.randn(latent_size).to(self.device)

        elif method == 'random_kdiffusion':
            sigmas = kwargs.get('sigmas', [14.6146])
            z = torch.randn(latent_size).to(self.device)
            z = z * (sigmas[0] ** 2 + 1) ** 0.5
        else:
            raise NotImplementedError

        return z.requires_grad_()

    def calculate_denoised(self, x: torch.Tensor, model_pred: torch.Tensor, alpha: torch.FloatTensor):
        """
        Compute Tweedie's formula in VP sampling.
        args:
            x (torch.Tensor): noisy sample
            model_pred (torch.Tensor): estimated noise
            alpha (torch.FloatTensor): alpha
        """
        return (x - (1-alpha).sqrt() * model_pred) / alpha.sqrt()

class Kdiffusion(StableDiffusion):
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        """
        Base class of LDM-based solvers based on (Karras et al 2022)
        Contain methods to leveraging VP diffusion model for VE sampling.
        For solvers like DPM and DPM++.

        args:
            solver_config (Dict): solver configurations (e.g. NFE)
            model_key (str): model key for loading pre-trained models
            device (torch.device): device
            **kwargs: additional arguments
        """
        super().__init__(solver_config, model_key, device, **kwargs)

        # load scheduler once again, not saved to self
        scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # convert alphas to sigmas (VP -> VE)
        total_sigmas = (1-scheduler.alphas_cumprod).sqrt() / scheduler.alphas_cumprod.sqrt()
        self.log_sigmas = total_sigmas.log()
        self.sigma_min, self.sigma_max = total_sigmas.min(), total_sigmas.max()

        # get karras sigmas
        self.k_sigmas = self.get_sigmas_karras(len(self.scheduler.timesteps), self.sigma_min, self.sigma_max)

    def get_sigmas_karras(self, n: int, sigma_min: float, sigma_max: float, rho: float=7., device: str='cpu'):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)

    def sigma_to_t(self, sigma: torch.FloatTensor):
        """Convert sigma to timestep. (find the closest index)"""
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def to_d(self, x: torch.Tensor, sigma: torch.FloatTensor, denoised: torch.Tensor):
        '''
        converts a denoiser output to a Karras ODE derivative
        args:
            x (torch.Tensor): noisy sample
            sigma (torch.FloatTensor): noise level
            denoised (torch.Tensor): denoised sample
        '''
        return (x - denoised) / sigma.item()
    
    def calculate_input(self, x: torch.Tensor, sigma: float):
        return x / (sigma ** 2 + 1) ** 0.5
    
    def calculate_denoised(self, x: torch.Tensor, model_pred: torch.Tensor, sigma: torch.FloatTensor):
        """
        Compute Tweedie's formula in VE sampling.
        args:
            x (torch.Tensor): noisy sample
            model_pred (torch.Tensor): estimated noise
            alpha (torch.FloatTensor): alpha
        """
        return x - model_pred * sigma
    
    def x_to_denoised(self, x, sigma, uc, c, cfg_guidance, t):
        """
        Get noisy sample and compute denoised samples.
        args:
            x (torch.Tensor): noisy sample
            sigma (float): noise level
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
            cfg_guidance (float): guidance scale
            t (torch.Tensor): timestep
        """
        xc = self.calculate_input(x, sigma)
        noise_uc, noise_c = self.predict_noise(xc, t, uc, c)
        noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
        denoised = self.calculate_denoised(x, noise_pred, sigma)
        uncond_denoised = self.calculate_denoised(x, noise_uc, sigma)
        return denoised, uncond_denoised


###########################################
# VP version samplers
###########################################

@register_solver("ddim")
class DDIM(StableDiffusion):
    """
    Basic DDIM solver for SD.
    VP sampling.
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = self.calculate_denoised(zt, noise_pred, at)

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
@register_solver("ddim_inversion")
class InversionDDIM(DDIM):
    """
    Reconstruction after inversion.
    Not for T2I generation.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img: torch.Tensor,
               cfg_guidance: float =7.5,
               prompt: Tuple[str]=["",""],
               callback_fn: Optional[Callable]=None,
               **kwargs):

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=c,
                                    cfg_guidance=cfg_guidance)
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = self.calculate_denoised(zt, noise_pred, at)

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver("ddim_edit")
class EditWordSwapDDIM(InversionDDIM):
    """
    Editing via WordSwap after inversion.
    Useful for text-guided image editing.
    Not for T2I generation.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img: torch.Tensor,
               cfg_guidance: float=7.5,
               prompt: Tuple[str]=["","",""],
               callback_fn: Optional[Callable]=None,
               **kwargs):

        # Text embedding
        uc, src_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, tgt_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=src_c,
                                    cfg_guidance=cfg_guidance)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DDIM-edit")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, tgt_c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = self.calculate_denoised(zt, noise_pred, at)

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
###########################################
# VE version samplers (K-diffusion)
###########################################

@register_solver("euler")
class EulerCFGSolver(Kdiffusion):
    """
    Karras Euler (VE casted)
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=self.k_sigmas).to(torch.float16)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="Euler")
        for i, _ in enumerate(pbar):
            sigma = self.k_sigmas[i]
            t = self.sigma_to_t(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, _ = self.x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
            
            d = self.to_d(x, sigma, denoised)
            # Euler method
            x = denoised + d * self.k_sigmas[i+1]

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(denoised)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("euler_a")
class EulerAncestralCFGSolver(Kdiffusion):
    """
    Karras Euler (VE casted) + Ancestral sampling
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=self.k_sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="Euler_a")
        for i, _ in enumerate(pbar):
            sigma = self.k_sigmas[i]
            t = self.sigma_to_t(sigma).to(self.device)
            sigma_down, sigma_up = get_ancestral_step(self.k_sigmas[i], self.k_sigmas[i + 1])
            with torch.no_grad():
                denoised, _ = self.x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
            
            # Euler method
            d = self.to_d(x, sigma, denoised)
            x = denoised + d * sigma_down
            
            if self.k_sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("dpm++_2s_a")
class DPMpp2sAncestralCFGSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=self.k_sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DPM++2s_a")
        for i, _ in enumerate(pbar):
            sigma = self.k_sigmas[i]
            t_1 = self.sigma_to_t(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, _ = self.x_to_denoised(x, sigma, uc, c, cfg_guidance, t_1)

            sigma_down, sigma_up = get_ancestral_step(self.k_sigmas[i], self.k_sigmas[i + 1])
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, self.k_sigmas[i], denoised)
                x = denoised + d * sigma_down
            else:
                # DPM-Solver++(2S)
                t, t_next = t_fn(self.k_sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                
                with torch.no_grad():
                    sigma_s = sigma_fn(s)
                    t_2 = self.sigma_to_t(sigma_s).to(self.device)
                    denoised_2, _ = self.x_to_denoised(x_2, sigma_s, uc, c, cfg_guidance, t_2)
                
                x = denoised_2 - torch.exp(-h) * denoised_2 + (sigma_fn(t_next) / sigma_fn(t)) * x

            # Noise addition
            if self.k_sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t_1, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("dpm++_2m")
class DPMpp2mCFGSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=self.k_sigmas).to(torch.float16)
        old_denoised = None # buffer
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DPM++_2m")
        for i, _ in enumerate(pbar):
            sigma = self.k_sigmas[i]
            t1 = self.sigma_to_t(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, _ = self.x_to_denoised(x, sigma, uc, c, cfg_guidance, t1)

            # solve ODE one step
            t, t_next = t_fn(self.k_sigmas[i]), t_fn(self.k_sigmas[i+1])
            h = t_next - t
            if old_denoised is None or self.k_sigmas[i+1] == 0:
                x = denoised + self.to_d(x, self.k_sigmas[i], denoised) * self.k_sigmas[i+1]
            else:
                h_last = t - t_fn(self.k_sigmas[i-1])
                r = h_last / h
                extra1 = -torch.exp(-h) * denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
                extra2 = torch.exp(-h) * x
                x = denoised + extra1 + extra2
            old_denoised = denoised

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t1, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


###########################################
# VP version samplers with CFG++
###########################################

@register_solver("ddim_cfg++")
class DDIMCFGpp(StableDiffusion):
    """
    DDIM solver for SD with CFG++.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = self.calculate_denoised(zt, noise_pred, at)

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
@register_solver("ddim_inversion_cfg++")
class InversionDDIMCFGpp(DDIMCFGpp):
    """
    Reconstruction after inversion.
    Not for T2I generation.
    """
    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float=1.0):

        # initialize z_0
        zt = z0.clone().to(self.device)

        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t-self.skip)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_uc) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img: torch.Tensor,
               cfg_guidance: float =7.5,
               prompt: Tuple[str]=["",""],
               callback_fn: Optional[Callable]=None,
               **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=c,
                                    cfg_guidance=cfg_guidance)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = self.calculate_denoised(zt, noise_pred, at)

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_edit_cfg++")
class EditWordSwapDDIMCFGpp(InversionDDIMCFGpp):
    """
    Editing via WordSwap after inversion.
    Useful for text-guided image editing.
    Not for T2I generation.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img: torch.Tensor,
               cfg_guidance: float=7.5,
               prompt: Tuple[str]=["","",""],
               callback_fn: Optional[Callable]=None,
               **kwargs):

        # Text embedding
        uc, src_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, tgt_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=src_c,
                                    cfg_guidance=cfg_guidance)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DDIM-edit")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, tgt_c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = self.calculate_denoised(zt, noise_pred, at)

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

###############################################
# VE version samplers (K-diffusion) with CFG++
###############################################

@register_solver("euler_cfg++")
class EulerCFGppSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=self.k_sigmas).to(torch.float16)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="Euler_cpp")
        for i, _ in enumerate(pbar):
            sigma = self.k_sigmas[i]
            t = self.sigma_to_t(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, uncond_denoised = self.x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
            
            d = self.to_d(x, sigma, uncond_denoised)
            # Euler method
            x = denoised + d * self.k_sigmas[i+1]

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("euler_a_cfg++")
class EulerAncestralCFGppSolver(Kdiffusion):
    """
    Karras Euler (VE casted) + Ancestral sampling
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=self.k_sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="Euler_a_cpp")
        for i, _ in enumerate(pbar):
            sigma = self.k_sigmas[i]
            t = self.sigma_to_t(sigma).to(self.device)
            sigma_down, sigma_up = get_ancestral_step(self.k_sigmas[i], self.k_sigmas[i + 1])
            with torch.no_grad():
                denoised, uncond_denoised = self.x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
            
            d = self.to_d(x, sigma, uncond_denoised)
            # Euler method
            x = denoised + d * sigma_down
            if self.k_sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("dpm++_2s_a_cfg++")
class DPMpp2sAncestralCFGppSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=self.k_sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DPM++2s_a_cpp")
        for i, _ in enumerate(pbar):
            sigma = self.k_sigmas[i]
            t_1 = self.sigma_to_t(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, uncond_denoised = self.x_to_denoised(x, sigma, uc, c, cfg_guidance, t_1)

            sigma_down, sigma_up = get_ancestral_step(self.k_sigmas[i], self.k_sigmas[i + 1])
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, self.k_sigmas[i], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                # DPM-Solver++(2S)
                t, t_next = t_fn(self.k_sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * uncond_denoised
                
                with torch.no_grad():
                    sigma_s = sigma_fn(s)
                    t_2 = self.sigma_to_t(sigma_s).to(self.device)
                    denoised_2, uncond_denoised_2 = self.x_to_denoised(x_2, sigma_s, uc, c, cfg_guidance, t_2)
                
                x = denoised_2 - torch.exp(-h) * uncond_denoised_2 + (sigma_fn(t_next) / sigma_fn(t)) * x
            # Noise addition
            if self.k_sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t_1, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    

@register_solver("dpm++_2m_cfg++")
class DPMpp2mCFGppSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=self.k_sigmas).to(torch.float16)
        old_denoised = None # buffer
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DPM++_2m_cpp")
        for i, _ in enumerate(pbar):
            sigma = self.k_sigmas[i]
            t_1 = self.sigma_to_t(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, uncond_denoised = self.x_to_denoised(x, sigma, uc, c, cfg_guidance, t_1)

            # solve ODE one step
            t, t_next = t_fn(self.k_sigmas[i]), t_fn(self.k_sigmas[i+1])
            h = t_next - t
            if old_denoised is None or self.k_sigmas[i+1] == 0:
                x = denoised + self.to_d(x, self.k_sigmas[i], uncond_denoised) * self.k_sigmas[i+1]
            else:
                h_last = t - t_fn(self.k_sigmas[i-1])
                r = h_last / h
                extra1 = -torch.exp(-h) * uncond_denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
                extra2 = torch.exp(-h) * x
                x = denoised + extra1 + extra2
            old_denoised = uncond_denoised

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t_1, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")

