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



### Achieved:
-  Have trained a ViT+CLIP late-interaction model, with LoRA on ViT, that:
    - maps image patches and text tokens into a shared space
    - uses MaxSim to compute fine-grained text–image similarity
and now, given a caption, correctly chooses its own image over several strong decoys — exactly what a retrieval head in a multimodal RAG system would do.


### Overview / Notebook Roadmap

| Cell | Purpose |
|------|---------|
| **1** | Import core libraries (PyTorch, timm, transformers, PEFT), configure CUDA. Defines project-root paths. |
| **2** | Download - Flickr8k dataset into a local disk layout (`images/` + `captions.json`). Verifies dataset integrity.
| **3** | Implement `ImageCaptionDataset` to load images and captions with minimal overhead, apply transforms. Consistent batching for both FT and evaluation. |
| **4** | Load a pretrained ViT (timm) and expose its patch-tokens (`196×768`) by **disabling global pooling.** Encodes each image into a *dense grid of patch embeddings*, preserving spatial structure for late interaction. |
| **5** | Load CLIP’s text encoder, project token embeddings into the ViT embedding space (`768-d`) via a learned projection head. This aligns the modalities so token–patch comparisons become geometrically meaningful. |
| **6** | Implement **MaxSim: a ColBERT-style late-interaction scorer** that performs **per-token max-patch alignment**. Produces a full `[B,B]` similarity matrix used for retrieval-style scoring and contrastive learning. |
| **7** | Define a **symmetric InfoNCE contrastive loss** coupling text→image and image→text directions. Loss encourages diagonal dominance and penalizes semantically inconsistent cross-modal matches. |
| **8** | Inject **LoRA adapters into the ViT attention layers** (`qkv`, `proj`) to enable **low-rank fine-tuning**. Only a small percentage of parameters are updated, preserving pretrained structure while adapting to Flickr captions. |
| **9** | Full training loop with **AdamW, LR scheduling, gradient clipping, and early stopping** for stable optimization. Computes patch–token alignment, updates LoRA+projection, and checkpoints the model every epoch. |
| **10** | Generate a loss-curve plot and save JSON logs. Stores “last”, “best”, and “final” adapter checkpoints. |
| **11** | Perform - qualitative retrieval sanity check by scoring a random batch with the trained model. **Confirms** late-interaction behavior by **verifying diagonal dominance** and performance (caption→image). |
| **12** | Perform - MaxSim Heatmaps. Visualizing this - builds a patch heatmap and plots it next to the original image. |





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

- Important: This lab runs heavily with the help of existing GPU, especially NVIDIA CPU. This is why pre-compiled versions of PyTorch to CUDA toolkits are being the main pin packages of this environment. And it makes the operations much easier. To do these on a different device, please investigate and either download CPU only Torch or download other wheels and builds which exactly run on your specific GPU. 