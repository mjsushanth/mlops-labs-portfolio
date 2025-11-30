### Lab: Vision Late-Interaction Retrieval with LoRA Fine-Tuning

### This lab demonstrates:
1. Explore images, local patch embeddings, and multi-vector retrieval.
2. (one encoder, one projection layer, one MaxSim function).
3. Train a tiny LoRA module on the vision encoder -- PEFT/Fine-Tuning/Quantization concepts.
4. Deeply understand:  projection heads, embedding normalization, batching, contrastive objectives, and PCA visualization of embedding geometry.
 
Most of this lab is inspired by deep learning research and trying to understand the libraries PyLate and ColPali. 
Re-implementing the core ideas that PyLate / ColPali use: PEFT, patchification, per-token text embeddings, parameter-efficient tuning.

### Algo:
- Patch embeddings from a Vision Transformer (timm).
- Token embeddings from a CLIP text encoder.
- MaxSim late-interaction scoring (token→patch similarity).
- LoRA fine-tuning applied to the vision encoder to adapt patch representations.
- Vision Encoder (timm): pre-pooling patch embeddings from a ViT model. 
  - `image → encoder → patch vectors (num_patches × dim)`
- Text Encoder (CLIP): `Encode text → token vectors, project them into same embedding space as image patches.`
- Late Interaction: `score(q, d) = Σ_i max_j( dot(q_i, d_j) )`
- Use a simple in-batch contrastive loss: `L = -log( exp(score_pos) / (exp(score_pos) + Σ_neg exp(score_neg)) )`
- LoRA Fine-Tuning: Apply LoRA to selected attention layers in the ViT encoder. Train only ~0.2–1.0% parameters.

#### About Data and Overview:
The entire lab runs on a small dataset (Flickr8k/COCO-1k subset).




### Instructions for users:


# ============================================================
# INSTALLATION - mamba. (preferred)
# ============================================================

```python

conda activate base
 cd .\Late_Interaction_MVR_PEFT_LoRA\

mamba env create -f environment.yml

conda activate vis_lateintr_peft_lora

python -m ipykernel install --user --name vis_lateintr_peft_lora --display-name "Python (vis_lateintr_peft_lora)"

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

```


```python
# ============================================================
# INSTALLATION for UV if needed.
# ============================================================
# 1. Create env:
#       uv venv weather_venv
#
# 2. Activate:
#       weather_venv\Scripts\activate          # Windows
#       source weather_venv/bin/activate        # Mac / Linux
#
# 3. Install:
#       uv pip install -r requirements.txt
```



#### More intuition about the Algorithm:

