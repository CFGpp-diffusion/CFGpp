# CFG++ : MANIFOLD-CONSTRAINED CLASSIFIER FREE GUIDANCE FOR DIFFUSION MODELS

![main figure](assets/main_figure.png)

---
## Summary

*Classifier-free guidance (CFG)* is a fundamental tool in modern diffusion models for text-guided generation. 

Although effective, CFG has notable drawbacks. For instance, 
- DDIM with CFG lacks invertibility, complicating image editing;
- Furthermore, high guidance scales, essential for high-quality outputs, frequently result in issues like mode collapse.

Contrary to the widespread belief that these are inherent limitations of diffusion models,
this paper reveals that the problems actually stem from the **off-manifold phenomenon associated with CFG**, rather than the diffusion models themselves.
 
Inspired by the recent advancements of diffusion model-based inverse problem solvers (DIS),  we reformulate text-guidance as an inverse problem with a text-conditioned score matching loss, and develop CFG++, a novel approach that tackles the off-manifold challenges inherent in traditional CFG. 

CFG++ features a surprisingly simple fix to CFG, yet it offers significant improvements. Furthermore, CFG++ enables seamless interpolation between unconditional and conditional sampling at lower guidance scales, consistently outperforming traditional CFG at all scales. 
CFG++ enables the reverse diffusion process can be understood as a reconstruction through scale-space representation.

## Setup

First, create your environment. We recommand to use the following comments. 

```
git clone https://github.com/CFGpp-diffusion/CFGpp.git
cd CFGpp
conda env create -f environment.yaml
```

For reproducability, using the same package version is neccessary since some dependencies lead to significant differences (for instance, diffusers). Nonetheless, improvement induced by CFG++ will be observed regardless the dependency.

If you run one of below examples, diffusers will automatically download checkpoints for SDv1.5 or SDXL.


## Examples

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
python -m examples.inversion --prompt "a photography of baby fox" --method "ddim" --cfg_guidance 7.5
```

- CFG ++
```
python -m examples.inversion --prompt "a photography of baby fox" --method "ddim_cfg++" --cfg_guidance 0.6
```

### Image Editing

- CFG
```
python -m examples.img_edit --src_prompt "a photography of baby fox" --tgt_prompt "a photography of a goat" --method "ddim" --cfg_guidance 7.5
```

- CFG ++
```
python -m examples.inversion --src_prompt "a photography of baby fox" --tgt_prompt "a photography of a goat" --method "ddim_cfg++" --cfg_guidance 0.6
```

## Reference
If you find our method is useful, please cite below or leave a star to this repository.

```
blahblah
```