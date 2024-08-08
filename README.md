# CFG++ : MANIFOLD-CONSTRAINED CLASSIFIER FREE GUIDANCE FOR DIFFUSION MODELS

This repository is the official implementation of [CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models](https://arxiv.org/abs/2406.08070v1), led by

[Hyungjin Chung*](https://www.hj-chung.com/), [Jeongsol Kim*](https://jeongsol.dev/), [Geon Yeong Park*](https://geonyeong-park.github.io/), [Hyelin Nam*](https://www.linkedin.com/in/hyelin-nam-01ab631a3/), [Jong Chul Ye](https://bispl.weebly.com/professor.html)

![main figure](assets/main_test_v5.png)

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://cfgpp-diffusion.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2311.18608-b31b1b.svg)](https://arxiv.org/abs/2406.08070)

---
## 🔥 Summary

*Classifier-free guidance (CFG)* is a fundamental tool in modern diffusion models for text-guided generation. Although effective, CFG requires high guidance scales, which has notable drawbacks:

1. **Mode collapse** and saturation
2. **Poor invertibility**
3. **Unnatural, curved PF-ODE trajectory**

We propose a simple fix to this seemingly inherent limitation and propose **CFG++** 🚀, which corrects the off-manifold problem of CFG. The following advantages are observed

1. **Small guidance scale** $\lambda \in$ [0, 1] can be used with a similar effect as $\omega \in$ [1.0, 12.5] in CFG
2. **Better sample quality** and better adherence to text
3. **Smooth, straighter** PF-ODE trajectory
4. **Enhanced invertibility**

Experimental results confirm that our method significantly enhances performance in **text-to-image generation, DDIM inversion, editing, and solving inverse problems**, suggesting a wide-ranging impact and potential applications in various fields that utilize text guidance.

## 🗓 ️News
- [20 Jul 2024] 🚨[Stable Diffusion WebUI reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) now supports CFG++. Thanks to the awesome work! Please checkout the [Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1e7enng/reforge_updates_new_samplers_new_scheduler_more/) for more details.
- [22 Jun 2024] 🚨[ComfyUI](https://openart.ai/workflows/dugumatai/new-sampler-euler_cfg/oGP4a011iYE2UpeTtXNH) now supports CFG++. Thanks to the awesome work of [@NotEvilGirl](https://gitea.com/NotEvilGirl/cfgpp)! For more details, please check out the [Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1dohy20/quick_overview_of_some_newish_stuff_in_comfyui/) and [Youtube video](https://www.youtube.com/watch?v=-GXJDz8i-Wo).
- [12 Jun 2024] Code and paper are uploaded.

## 🛠️ Setup
First, create your environment. We recommend using the following comments. 

```
git clone https://github.com/CFGpp-diffusion/CFGpp.git
cd CFGpp
conda env create -f environment.yaml
```

For reproducibility, using the same package version is necessary since some dependencies lead to significant differences (for instance, diffusers). Nonetheless, improvement induced by CFG++ will be observed regardless the dependency.

If you run one of the below examples, diffusers will automatically download checkpoints for SDv1.5 or SDXL.


## 🌄 Examples

### Text-to-Image generation

- CFG
```
python -m examples.text_to_img --prompt "a portrait of a dog" --method "ddim" --cfg_guidance 7.5
```

- CFG ++
```
python -m examples.text_to_img --prompt "a portrait of a dog" --method "ddim_cfg++" --cfg_guidance 0.6
```

### Image Inversion

- CFG
```
python -m examples.inversion --prompt "a photography of baby fox" --method "ddim_inversion" --cfg_guidance 7.5
```

- CFG ++
```
python -m examples.inversion --prompt "a photography of baby fox" --method "ddim_inversion_cfg++" --cfg_guidance 0.6
```

> [!tip]
> If you want to use SDXL, add ``--model sdxl``.

## 🔬 Callback

We provide callback functionality to monitor intermediate samples during the diffusion reverse process. For now, the function could be called only at the end of each timestep, for the readability of scripts.

Currently, we provide two options (default: None).
- draw_tweedie : save $\hat x_{0|t}$ to workdir
- draw_noisy : save $x_t$ to workdir

Note that using callback may take more time due to file save. You can refer utils/callback_util.py for details.

## 📝 Citation
If you find our method useful, please cite as below or leave a star to this repository.

```
@article{chung2024cfg++,
  title={CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models},
  author={Chung, Hyungjin and Kim, Jeongsol and Park, Geon Yeong and Nam, Hyelin and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2406.08070},
  year={2024}
}
```

> [!note]
> This work is currently in the preprint stage, and there may be some changes to the code.
