### MLops - Lab Submission.
#### Course IE 7374 - MLOps, Northeastern University.

- Developing this lab user ModelDevelopment/ -- concepts. ( Feat Selection/Tuning/Distillation/Quantization/Pruning etc.)
- I'm gonna add a twist to this and choose new concepts; around `PyLate + FAISS + ColBERT` and `LEANN` 
- Goal: Explore Multi-hop reasoning, Neural aggregation, Evidence composition, finetuning
- This is to explore latest research like MuVera, DeepMind RETRO, GNN-style evidence modules.

#### Dataset/Models:
- HotpotQA (multi-hop, distractors, supporting facts)
  - **HotpotQA includes irrelevant paragraphs intentionally.**
- PyLate (ColBERT-style multi-vector retrieval + finetuning)
  - High-recall retrieval + multi-vector scoring + triplet finetuning.
  - ( ColBERT token-level representations, + powerful retrieval over multi-hop queries, + reranking, finetuning )
- LEANN (lightweight neural attentive network + aggregator + multi-step reasoning)
  - `https://github.com/yichuan-w/LEANN`, LEANN = Lightweight Evidence Aggregation Neural Network
  - neural module that takes multiple retrieved pieces, assign weights / relevance, **learns to aggregate them**.
  - LEANN tries to do single unified evidence representation. It is not a generator. It is a reasoning layer.
  - Sits between retrieval and answering.

#### Pipeline:
```
Query
  ↓
PyLate (ColBERT)
  ↓ top 20 passages
Rerank (PyLate late-interaction)
  ↓ top 8
LEANN Evidence Aggregator
  ↓ combines supporting facts into a single representation
Lightweight Answer Head
  ↓
Answer / Classification / Short Span Extraction
```

#### Tools:
- Python 3.12, UV, Stuff mentioned above.


1. 01_hotpot_prep.ipynb — Data download + small subset 
2. 02_pylate_baseline.ipynb — Build index + baseline retrieval
3. 03_pylate_finetune.ipynb — Finetune + rebuild + re-evaluate
4. 04_leann_aggregation.ipynb — Minimal LEANN-style aggregator (depending on time.)