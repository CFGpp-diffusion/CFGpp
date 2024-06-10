# CFG++ : MANIFOLD-CONSTRAINED CLASSIFIER FREE GUIDANCE FOR DIFFUSION MODELS


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
