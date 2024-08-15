"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""

from typing import Any, Callable, Dict, Optional

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


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n+1, device=device)[:-1]
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

########################

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        self.device = device

        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.total_alphas = self.scheduler.alphas_cumprod.clone()
        
        self.sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        self.log_sigmas = self.sigmas.log()
        
        total_timesteps = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Solver must implement sample() method.")

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def get_text_embed(self, null_prompt, prompt):
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

    def encode(self, x):
        """
        xt -> zt
        """
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, zt):
        """
        zt -> xt
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
                          **kwargs):
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
            size = kwargs.get('latent_dim', (1, 4, 64, 64))
            z = torch.randn(size).to(self.device)
        elif method == 'random_kdiffusion':
            size = kwargs.get('latent_dim', (1, 4, 64, 64))
            sigmas = kwargs.get('sigmas', [14.6146])
            z = torch.randn(size).to(self.device)
            z = z * (sigmas[0] ** 2 + 1) ** 0.5
        else:
            raise NotImplementedError

        return z.requires_grad_()
    
    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def to_d(self, x, sigma, denoised):
        '''converts a denoiser output to a Karras ODE derivative'''
        return (x - denoised) / sigma.item()
    
    def get_ancestral_step(self, sigma_from, sigma_to, eta=1.):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        if not eta:
            return sigma_to, 0.
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up
    
    def calculate_input(self, x, sigma):
        return x / (sigma ** 2 + 1) ** 0.5
    
    def calculate_denoised(self, x, model_pred, sigma):
        return x - model_pred * sigma
    
    def kdiffusion_x_to_denoised(self, x, sigma, uc, c, cfg_guidance, t):
        xc = self.calculate_input(x, sigma)
        noise_uc, noise_c = self.predict_noise(xc, t, uc, c)
        noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
        denoised = self.calculate_denoised(x, noise_pred, sigma)
        uncond_denoised = self.calculate_denoised(x, noise_uc, sigma)
        return denoised, uncond_denoised

###########################################
# Base version
###########################################

@register_solver("ddim")
class BaseDDIM(StableDiffusion):
    """
    Basic DDIM solver for SD.
    Useful for text-to-image generation
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """

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
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

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
    
    
@register_solver("euler")
class EulerCFGSolver(StableDiffusion):
    """
    Karras Euler (VE casted)
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # perpare alphas and sigmas
        timesteps = reversed(torch.linspace(0, 1000, len(self.scheduler.timesteps)+1).long())
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            t = self.timestep(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
            
            d = self.to_d(x, sigma, denoised)
            # Euler method
            x = denoised + d * sigmas[i+1]

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(denoised)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("euler_a")
class EulerAncestralCFGSolver(StableDiffusion):
    """
    Karras Euler (VE casted) + Ancestral sampling
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            t = self.timestep(sigma).to(self.device)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
            
            # Euler method
            d = self.to_d(x, sigma, denoised)
            x = denoised + d * sigma_down
            
            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)

        # for the last step, do not add noise
        img = self.decode(denoised)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("dpm++_2s_a")
class DPMpp2sAncestralCFGSolver(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, sigmas[i], denoised)
                x = denoised + d * sigma_down
            else:
                # DPM-Solver++(2S)
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                
                with torch.no_grad():
                    sigma_s = sigma_fn(s)
                    t_2 = self.timestep(sigma_s).to(self.device)
                    denoised_2, _ = self.kdiffusion_x_to_denoised(x_2, sigma_s, uc, c, cfg_guidance, t_2)
                
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
            # Noise addition
            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("dpm++_2m")
class DPMpp2mCFGSolver(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)
        old_denoised = None # buffer
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            # solve ODE one step
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i+1])
            h = t_next - t
            if old_denoised is None or sigmas[i+1] == 0:
                x = denoised + self.to_d(x, sigmas[i], denoised) * sigmas[i+1]
            else:
                h_last = t - t_fn(sigmas[i-1])
                r = h_last / h
                extra1 = -torch.exp(-h) * denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
                extra2 = torch.exp(-h) * x
                x = denoised + extra1 + extra2
            old_denoised = denoised

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver("ddim_inversion")
class InversionDDIM(BaseDDIM):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
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
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

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
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
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
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

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
# CFG++ version
###########################################

@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DDIM solver for SD with CFG++.
    Useful for text-to-image generation
    """
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """

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
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

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
    
    
@register_solver("euler_cfg++")
class EulerCFGppSolver(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # perpare alphas and sigmas
        timesteps = reversed(torch.linspace(0, 1000, len(self.scheduler.timesteps)+1).long())
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            t = self.timestep(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, uncond_denoised = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
            
            d = self.to_d(x, sigma, uncond_denoised)
            # Euler method
            x = denoised + d * sigmas[i+1]

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(denoised)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("euler_a_cfg++")
class EulerAncestralCFGppSolver(StableDiffusion):
    """
    Karras Euler (VE casted) + Ancestral sampling
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            t = self.timestep(sigma).to(self.device)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            with torch.no_grad():
                denoised, uncond_denoised = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
            
            d = self.to_d(x, sigma, uncond_denoised)
            # Euler method
            x = denoised + d * sigma_down
            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)

        # for the last step, do not add noise
        img = self.decode(denoised)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("dpm++_2s_a_cfg++")
class DPMpp2sAncestralCFGppSolver(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, uncond_denoised = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, sigmas[i], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                # DPM-Solver++(2S)
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * uncond_denoised
                
                with torch.no_grad():
                    sigma_s = sigma_fn(s)
                    t_2 = self.timestep(sigma_s).to(self.device)
                    denoised_2, uncond_denoised_2 = self.kdiffusion_x_to_denoised(x_2, sigma_s, uc, c, cfg_guidance, t_2)
                
                x = denoised_2 - torch.exp(-h) * uncond_denoised_2 + (sigma_fn(t_next) / sigma_fn(t)) * x
            # Noise addition
            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    

@register_solver("dpm++_2m_cfg++")
class DPMpp2mCFGppSolver(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)
        old_denoised = None # buffer
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, uncond_denoised = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            # solve ODE one step
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i+1])
            h = t_next - t
            if old_denoised is None or sigmas[i+1] == 0:
                x = denoised + self.to_d(x, sigmas[i], uncond_denoised) * sigmas[i+1]
            else:
                h_last = t - t_fn(sigmas[i-1])
                r = h_last / h
                extra1 = -torch.exp(-h) * uncond_denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
                extra2 = torch.exp(-h) * x
                x = denoised + extra1 + extra2
            old_denoised = uncond_denoised

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver("ddim_inversion_cfg++")
class InversionDDIMCFGpp(BaseDDIMCFGpp):
    """
    Editing via WordSwap after inversion.
    Useful for text-guided image editing.
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
               src_img,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
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
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

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
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
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
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

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


#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")

