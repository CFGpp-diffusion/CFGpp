"""
This module is for sampling with SDXL.
"""


import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from diffusers import (AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler,
                       StableDiffusionXLPipeline, UNet2DConditionModel)
from diffusers.models.attention_processor import (AttnProcessor2_0,
                                                  LoRAAttnProcessor2_0,
                                                  LoRAXFormersAttnProcessor,
                                                  XFormersAttnProcessor)
from safetensors.torch import load_file
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

class SDXL():
    def __init__(self, 
                 solver_config: Dict,
                 model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
                 dtype=torch.float16,
                 device='cuda'):
        """
        The base class of LDM-based solvers for VP sampling, especially for SDXL.
        We load pre-trained VAE, text-encoder, and U-Net models from diffusers.
        Also, compute pre-defined coefficients.

        args:
            solver_config (Dict): solver configurations (e.g. NFE)
            model_key (str): model key for loading pre-trained models
            device (torch.device): device
            **kwargs: additional arguments
        """
        # pre-trained model loading
        self.device = device
        self.dtype = dtype
        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.unet = pipe.unet

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size
        
        # load scheduler
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        # time discretization
        self.total_alphas = self.scheduler.alphas_cumprod.clone()
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def _text_embed(self, prompt: str, tokenizer: Callable, text_enc: Callable, clip_skip: bool):
        """
        embedding function of SDXL
        args:
            prompt (str): text prompt
            tokenizer (Callable): loaded tokenizer
            text_enc (Callable): loaded text encoder
            clip_skip (bool): 
        """
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)

        pool_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # +2 because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        return prompt_embeds, pool_prompt_embeds

    @torch.no_grad()
    def get_text_embed(self,
                       null_prompt_1: str, 
                       prompt_1: str,
                       null_prompt_2: Optional[str]=None,
                       prompt_2: Optional[str]=None,
                       clip_skip: Optional[bool]=None):
        '''
        Get text embedding.
        
        TODO:
        At this time, assume that batch_size = 1.
        We should extend the code to batch_size > 1.

        args:
            null_prompt_1 (str): null text for text encoder 1
            prompt_1 (str): guidance text for text encoder 1
            null_prompt_2 (Optional[str]): null prompt for text encoder 2. None: use null_prompt_1.
            prompt_2 (Optional[str]): prompt for text encoder 2. None: use prompt_1.
            clip_skip (Optional[bool]):
        '''        
        # Encode the prompts
        # if prompt_2 is None, set same as prompt_1
        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1


        prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if prompt_2 is None:
            prompt_embed = [prompt_embed_1]
        else:
            # Comment on diffusers' source code:
            # "We are only ALWAYS interested in the pooled output of the final text encoder"
            # i.e. we overwrite the pool_prompt_embed with the new one
            prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            prompt_embed = [prompt_embed_1, prompt_embed_2]
        
        null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if null_prompt_2 is None:
            null_embed = [null_embed_1]
        else:
            null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            null_embed = [null_embed_1, null_embed_2]

        # concat embeds from two encoders
        null_prompt_embeds = torch.concat(null_embed, dim=-1)
        prompt_embeds = torch.concat(prompt_embed, dim=-1)

        return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed            

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """
        Encode image to latent features.
        args:
            x (torch.Tensor): image
        """
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor 

    # @torch.no_grad() 
    def decode(self, zt: torch.Tensor):
        """
        Decode latent features to image.
        args:
            zt (torch.Tensor): latent
        """
        # make sure the VAE is in float32 mode, as it overflows in float16
        # needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        # if needs_upcasting:
        #     self.upcast_vae()
        #     zt = zt.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()
        return image


    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: Optional[torch.Tensor],
                      c: Optional[torch.Tensor],
                      added_cond_kwargs: Optional[Dict]):

        t_in = t.unsqueeze(0)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    def _get_add_time_ids(self,
                          original_size: Tuple[int, int],
                          crops_coords_top_left: Tuple[int, int],
                          target_size: Tuple[int, int],
                          dtype: str,
                          text_encoder_projection_dim: int):
        """
        Create additional kwargs for SDXL.
        Taken from diffusers pipeline. 
        args:
            original_size (Tuple[int, int]): original size of the image
            crops_coords_top_left (Tuple[int, int]): top-left coordinates of the crop
            target_size (Tuple[int, int]): target size of the image
            dtype (str): data type
            text_encoder_projection_dim (int): projection dimension of the text encoder
        """
        add_time_ids = list(original_size+crops_coords_top_left+target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        assert expected_add_embed_dim == passed_add_embed_dim, (
             f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               prompt1 = ["", ""],
               prompt2 = ["", ""],
               cfg_guidance:float=5.0,
               original_size: Optional[Tuple[int, int]]=None,
               crops_coords_top_left: Tuple[int, int]=(0, 0),
               target_size: Optional[Tuple[int, int]]=None,
               negative_original_size: Optional[Tuple[int, int]]=None,
               negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
               negative_target_size: Optional[Tuple[int, int]]=None,
               clip_skip: Optional[int]=None,
               **kwargs):

        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # embedding
        (null_prompt_embeds,
         prompt_embeds,
         pool_null_embed,
         pool_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)

        # prepare kwargs for SDXL
        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed 

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            # do cfg
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        # reverse sampling
        zt = self.reverse_process(null_prompt_embeds, prompt_embeds, cfg_guidance, add_cond_kwargs, target_size, **kwargs)

        # decode
        with torch.no_grad():
            img = self.decode(zt)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          latent_size: tuple=(1, 4, 128, 128),
                          add_cond_kwargs: Optional[dict]=None,
                          **kwargs):
        """
        Initialize latent features.
        Simply, sample from Gaussian distribution or do inversion.
        args:
            method (str): initialization method
            src_img (torch.Tensor): source image
            add_cond_kwargs (dict): additional conditional arguments
            **kwargs: additional arguments
        """
        if method == 'ddim':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               kwargs.get('cfg_guidance', 0.0),
                               add_cond_kwargs)
        elif method == 'npi':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               1.0,
                               add_cond_kwargs)
        elif method == 'random':
            z = torch.randn(latent_size).to(self.device)
        elif method == "random_kdiffusion":
            sigmas = kwargs.get('sigmas', [14.6146])
            z = torch.randn(latent_size).to(self.device)
            z = z * (sigmas[0] ** 2 + 1) ** 0.5
        else: 
            raise NotImplementedError

        return z.requires_grad_()

    def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
        # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted. 
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt
    
    def reverse_process(self, *args, **kwargs):
        raise NotImplementedError

    def calculate_denoised(self, x: torch.Tensor, model_pred: torch.Tensor, alpha: torch.FloatTensor):
        """
        Compute Tweedie's formula in VP sampling.
        args:
            x (torch.Tensor): noisy sample
            model_pred (torch.Tensor): estimated noise
            alpha (torch.FloatTensor): alpha
        """
        return (x - (1-alpha).sqrt() * model_pred) / alpha.sqrt()

class SDXLLightning(SDXL):
    def __init__(self, 
                 solver_config: dict,
                 base_model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
                 light_model_ckpt:str="ckpt/sdxl_lightning_4step_unet.safetensors",
                 dtype=torch.float16,
                 device='cuda'):

        self.device = device

        # load the student model
        unet = UNet2DConditionModel.from_config(base_model_key, subfolder="unet").to("cuda", torch.float16)
        ext = os.path.splitext(light_model_ckpt)[1]
        if ext == ".safetensors":
            state_dict = load_file(light_model_ckpt)
        else:
            state_dict = torch.load(light_model_ckpt, map_location="cpu")
        print(unet.load_state_dict(state_dict, strict=True))
        unet.requires_grad_(False)
        self.unet = unet

        #pipe2 = StableDiffusionXLPipeline.from_single_file(light_model_ckpt, torch_dtype=dtype).to(device)
        pipe = StableDiffusionXLPipeline.from_pretrained(base_model_key, unet=self.unet, torch_dtype=dtype).to(device)
        self.dtype = dtype

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        # sampling parameters
        self.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        self.total_alphas = self.scheduler.alphas_cumprod.clone()
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        #self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)


class Kdiffusion(SDXL):
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
                 dtype=torch.float16,
                 device: Optional[torch.device]=None,
                 **kwargs):
        """
        Base SDXL class of LDM-based solvers based on (Karras et al 2022)
        Contain methods to leveraging VP diffusion model for VE sampling.
        For solvers like DPM and DPM++.

        args:
            solver_config (Dict): solver configurations (e.g. NFE)
            model_key (str): model key for loading pre-trained models
            device (torch.device): device
            **kwargs: additional arguments
        """
        super().__init__(solver_config, model_key, dtype, device, **kwargs)

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
    
    def x_to_denoised(self, x, sigma, uc, c, cfg_guidance, t, added_cond_kwargs):
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
        noise_uc, noise_c = self.predict_noise(xc, t, uc, c, added_cond_kwargs)
        noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
        denoised = self.calculate_denoised(x, noise_pred, sigma)
        uncond_denoised = self.calculate_denoised(x, noise_uc, sigma)
        return denoised, uncond_denoised

###########################################
# VP version samplers
###########################################

@register_solver('ddim')
class DDIM(SDXL):
    """
    Basic DDIM solver for SDXL.
    VP sampling.
    """
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = { 'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t


@register_solver("ddim_edit")
class EditWardSwapDDIM(DDIM):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               prompt1 = ["", "", ""],
               prompt2 = ["", "", ""],
               cfg_guidance:float=5.0,
               original_size: Optional[Tuple[int, int]]=None,
               crops_coords_top_left: Tuple[int, int]=(0, 0),
               target_size: Optional[Tuple[int, int]]=None,
               negative_original_size: Optional[Tuple[int, int]]=None,
               negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
               negative_target_size: Optional[Tuple[int, int]]=None,
               clip_skip: Optional[int]=None,
               **kwargs):

        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # embedding
        (null_prompt_embeds,
         src_prompt_embeds,
         pool_null_embed,
         pool_src_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)

        (_,
         tgt_prompt_embeds,
         _,
         pool_tgt_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[2], prompt2[0], prompt2[2], clip_skip)

        # prepare kwargs for SDXL
        add_src_text_embeds = pool_src_prompt_embed
        add_tgt_text_embeds = pool_tgt_prompt_embed

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=src_prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_src_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=src_prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_src_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed 

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            # do cfg
            add_src_text_embeds = torch.cat([negative_text_embeds, add_src_text_embeds], dim=0)
            add_tgt_text_embeds = torch.cat([negative_text_embeds, add_tgt_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_src_cond_kwargs = {
            'text_embeds': add_src_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        add_tgt_cond_kwargs = {
            'text_embeds': add_tgt_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        # reverse sampling
        zt = self.reverse_process(null_prompt_embeds,
                                  src_prompt_embeds, 
                                  tgt_prompt_embeds,
                                  cfg_guidance,
                                  add_src_cond_kwargs,
                                  add_tgt_cond_kwargs,
                                  **kwargs)

        # decode
        with torch.no_grad():
            img = self.decode(zt)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    def reverse_process(self,
                        null_prompt_embeds,
                        src_prompt_embeds,
                        tgt_prompt_embed,
                        cfg_guidance,
                        add_src_cond_kwargs,
                        add_tgt_cond_kwargs,
                        callback_fn=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=kwargs.get('src_img', None),
                                    uc=null_prompt_embeds,
                                    c=src_prompt_embeds,
                                    cfg_guidance=cfg_guidance,
                                    add_cond_kwargs=add_src_cond_kwargs)

        # sampling
        pbar = tqdm(self.scheduler.timesteps, desc='SDXL')
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_next = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, 
                                                       null_prompt_embeds,
                                                       tgt_prompt_embed,
                                                       add_tgt_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t

###############################################
# SDXL Lightning with CFG++
###############################################
@register_solver('ddim_lightning')
class BaseDDIMLight(DDIM, SDXLLightning):
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
    
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return super().reverse_process(null_prompt_embeds, 
                                        prompt_embeds, 
                                        cfg_guidance, 
                                        add_cond_kwargs, 
                                        shape, 
                                        callback_fn, 
                                        **kwargs)

###########################################
# VE version samplers (K-diffusion)
###########################################

@register_solver("euler")
class EulerCFGSolver(Kdiffusion):
    """
    Karras Euler (VE casted)
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        x = self.initialize_latent(method="random_kdiffusion",
                                    size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL-Euler')
        for step, _ in enumerate(pbar):
            sigma = self.k_sigmas[step]
            t = self.sigma_to_t(sigma).to(self.device)

            with torch.no_grad():
                denoised, _ = self.x_to_denoised(x,
                                                 sigma,
                                                 null_prompt_embeds,
                                                 prompt_embeds,
                                                 cfg_guidance,
                                                 t,
                                                 add_cond_kwargs)

            d = self.to_d(x, sigma, denoised)
            # Euler method
            x = denoised + d * self.k_sigmas[step+1]

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return x


@register_solver("euler_a")
class EulerAncestralCFGSolver(Kdiffusion):
    """
    Karras Euler (VE casted) + Ancestral sampling
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        x = self.initialize_latent(method="random_kdiffusion",
                                    size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL-Euler_a')
        for step, _ in enumerate(pbar):
            sigma = self.k_sigmas[step]
            t = self.sigma_to_t(sigma).to(self.device)
            sigma_down, sigma_up = get_ancestral_step(self.k_sigmas[step], self.k_sigmas[step+1])
            with torch.no_grad():
                denoised, _ = self.x_to_denoised(x,
                                                 sigma,
                                                 null_prompt_embeds,
                                                 prompt_embeds,
                                                 cfg_guidance,
                                                 t,
                                                 add_cond_kwargs)
            # Euler method
            d = self.to_d(x, sigma, denoised)
            x = denoised + d * sigma_down

            if self.k_sigmas[step + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return x


@register_solver("dpm++_2s_a")
class DPMpp2AncestralCFGSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()

        x = self.initialize_latent(method="random_kdiffusion",
                                    size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL-DPM++2s_a')
        for step, _ in enumerate(pbar):
            sigma = self.k_sigmas[step]
            t_1 = self.sigma_to_t(sigma).to(self.device)
            with torch.no_grad():
                denoised, _ = self.x_to_denoised(x,
                                                 sigma,
                                                 null_prompt_embeds,
                                                 prompt_embeds,
                                                 cfg_guidance,
                                                 t_1,
                                                 add_cond_kwargs)

            sigma_down, sigma_up = get_ancestral_step(self.k_sigmas[step], self.k_sigmas[step + 1])
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, self.k_sigmas[step], denoised)
                x = denoised + d * sigma_down
            else:
                # DPM-Solver++(2S)
                t, t_next = t_fn(self.k_sigmas[step]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                
                with torch.no_grad():
                    sigma_s = sigma_fn(s)
                    t_2 = self.sigma_to_t(sigma_s).to(self.device)
                    denoised_2, _ = self.x_to_denoised(x_2,
                                                       sigma_s,
                                                       null_prompt_embeds,
                                                       prompt_embeds,
                                                       cfg_guidance,
                                                       t_2,
                                                       add_cond_kwargs)
                
                x = denoised_2 - torch.exp(-h) * denoised_2 + (sigma_fn(t_next) / sigma_fn(t)) * x

            # Noise addition
            if self.k_sigmas[step + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return x

@register_solver("dpm++_2m")
class DPMpp2mCFGSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()

        x = self.initialize_latent(method="random_kdiffusion",
                                    size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        old_denoised=None # buffer
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL-DPM++2m')
        for step, _ in enumerate(pbar):
            sigma = self.k_sigmas[step]
            t_1 = self.sigma_to_t(sigma).to(self.device)
            with torch.no_grad():
                denoised, _ = self.x_to_denoised(x,
                                                 sigma,
                                                 null_prompt_embeds,
                                                 prompt_embeds,
                                                 cfg_guidance,
                                                 t_1,
                                                 add_cond_kwargs)
            # solve ODE one step
            t, t_next = t_fn(self.k_sigmas[step]), t_fn(self.k_sigmas[step+1])
            h = t_next - t
            if old_denoised is None or self.k_sigmas[step+1] == 0:
                x = denoised + self.to_d(x, self.k_sigmas[step], denoised) * self.k_sigmas[step+1]
            else:
                h_last = t - t_fn(self.k_sigmas[step-1])
                r = h_last / h
                extra1 = -torch.exp(-h) * denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
                extra2 = torch.exp(-h) * x
                x = denoised + extra1 + extra2
            old_denoised = denoised

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return x

###########################################
# VP version samplers with CFG++
###########################################

@register_solver("ddim_cfg++")
class DDIMCFGpp(SDXL):
    """
    DDIM solver for SDXL with CFG++.
    """
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = { 'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t

@register_solver("ddim_edit_cfg++")
class EditWardSwapDDIMCFGpp(EditWardSwapDDIM):
    @torch.no_grad()
    def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
        # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted. 
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_uc) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt

    def reverse_process(self,
                        null_prompt_embeds,
                        src_prompt_embeds,
                        tgt_prompt_embed,
                        cfg_guidance,
                        add_src_cond_kwargs,
                        add_tgt_cond_kwargs,
                        callback_fn=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=kwargs.get('src_img', None),
                                    uc=null_prompt_embeds,
                                    c=src_prompt_embeds,
                                    cfg_guidance=cfg_guidance,
                                    add_cond_kwargs=add_src_cond_kwargs)

        # sampling
        pbar = tqdm(self.scheduler.timesteps, desc='SDXL')
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_next = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, 
                                                       null_prompt_embeds,
                                                       tgt_prompt_embed,
                                                       add_tgt_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t


###############################################
# SDXL Lightning with CFG++
###############################################

@register_solver('ddim_cfg++_lightning')
class BaseDDIMCFGppLight(DDIMCFGpp, SDXLLightning):
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
    
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return super().reverse_process(null_prompt_embeds, 
                                        prompt_embeds, 
                                        cfg_guidance, 
                                        add_cond_kwargs, 
                                        shape, 
                                        callback_fn, 
                                        **kwargs)


###############################################
# VE version samplers (K-diffusion) with CFG++
###############################################

@register_solver("euler_cfg++")
class EulerCFGppSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        x = self.initialize_latent(method="random_kdiffusion",
                                    size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL-Euler')
        for step, _ in enumerate(pbar):
            sigma = self.k_sigmas[step]
            t = self.sigma_to_t(sigma).to(self.device)

            with torch.no_grad():
                denoised, uncond_denoised = self.x_to_denoised(x,
                                                 sigma,
                                                 null_prompt_embeds,
                                                 prompt_embeds,
                                                 cfg_guidance,
                                                 t,
                                                 add_cond_kwargs)

            d = self.to_d(x, sigma, uncond_denoised)
            # Euler method
            x = denoised + d * self.k_sigmas[step+1]

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return x


@register_solver("euler_a_cfg++")
class EulerAncestralCFGppSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        x = self.initialize_latent(method="random_kdiffusion",
                                    size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL-Euler_a')
        for step, _ in enumerate(pbar):
            sigma = self.k_sigmas[step]
            t = self.sigma_to_t(sigma).to(self.device)
            sigma_down, sigma_up = get_ancestral_step(self.k_sigmas[step], self.k_sigmas[step+1])
            with torch.no_grad():
                denoised, uncond_denoised = self.x_to_denoised(x,
                                                 sigma,
                                                 null_prompt_embeds,
                                                 prompt_embeds,
                                                 cfg_guidance,
                                                 t,
                                                 add_cond_kwargs)
            # Euler method
            d = self.to_d(x, sigma, uncond_denoised)
            x = denoised + d * sigma_down

            if self.k_sigmas[step + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return x

@register_solver("dpm++_2s_a_cfg++")
class DPMpp2sAncestralCFGppSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()

        x = self.initialize_latent(method="random_kdiffusion",
                                    size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL-DPM++2s_a')
        for step, _ in enumerate(pbar):
            sigma = self.k_sigmas[step]
            t_1 = self.sigma_to_t(sigma).to(self.device)
            with torch.no_grad():
                denoised, uncond_denoised = self.x_to_denoised(x,
                                                 sigma,
                                                 null_prompt_embeds,
                                                 prompt_embeds,
                                                 cfg_guidance,
                                                 t_1,
                                                 add_cond_kwargs)

            sigma_down, sigma_up = get_ancestral_step(self.k_sigmas[step], self.k_sigmas[step + 1])
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, self.k_sigmas[step], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                # DPM-Solver++(2S)
                t, t_next = t_fn(self.k_sigmas[step]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * uncond_denoised
                
                with torch.no_grad():
                    sigma_s = sigma_fn(s)
                    t_2 = self.sigma_to_t(sigma_s).to(self.device)
                    denoised_2, uncond_denoised_2 = self.x_to_denoised(x_2,
                                                       sigma_s,
                                                       null_prompt_embeds,
                                                       prompt_embeds,
                                                       cfg_guidance,
                                                       t_2,
                                                       add_cond_kwargs)
                
                x = denoised_2 - torch.exp(-h) * uncond_denoised_2 + (sigma_fn(t_next) / sigma_fn(t)) * x

            # Noise addition
            if self.k_sigmas[step + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return x


@register_solver("dpm++_2m_cfg++")
class DPMpp2mCFGppSolver(Kdiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()

        x = self.initialize_latent(method="random_kdiffusion",
                                    size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        old_denoised=None # buffer
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL-DPM++2m')
        for step, _ in enumerate(pbar):
            sigma = self.k_sigmas[step]
            t_1 = self.sigma_to_t(sigma).to(self.device)
            with torch.no_grad():
                denoised, uncond_denoised = self.x_to_denoised(x,
                                                 sigma,
                                                 null_prompt_embeds,
                                                 prompt_embeds,
                                                 cfg_guidance,
                                                 t_1,
                                                 add_cond_kwargs)
            # solve ODE one step
            t, t_next = t_fn(self.k_sigmas[step]), t_fn(self.k_sigmas[step+1])
            h = t_next - t
            if old_denoised is None or self.k_sigmas[step+1] == 0:
                x = denoised + self.to_d(x, self.k_sigmas[step], uncond_denoised) * self.k_sigmas[step+1]
            else:
                h_last = t - t_fn(self.k_sigmas[step-1])
                r = h_last / h
                extra1 = -torch.exp(-h) * uncond_denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
                extra2 = torch.exp(-h) * x
                x = denoised + extra1 + extra2
            old_denoised = uncond_denoised

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return x

@register_solver('dpm++_2m_cfg++_lightning')
class DPMpp2mCFGppLightningSolver(DPMpp2mCFGppSolver, SDXLLightning):
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
    
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return super().reverse_process(null_prompt_embeds, 
                                        prompt_embeds, 
                                        cfg_guidance, 
                                        add_cond_kwargs, 
                                        shape, 
                                        callback_fn, 
                                        **kwargs)

#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")
        
