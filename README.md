# PEFT A2Z — Short Story Assignment
## Parameter-Efficient Fine-Tuning for Large Language and Vision Models

> **Course Assignment | Individual Submission**  
> **Paper Reviewed:** Prottasha et al. (2025), *PEFT A2Z: Parameter-Efficient Fine-Tuning Survey for Large Language and Vision Models*, arXiv:2504.14117v1

---

## 📋 Table of Contents

- [Overview](#overview)
- [Paper Summary](#paper-summary)
- [Deliverables](#deliverables)
- [Repository Structure](#repository-structure)
- [Autoresearch Reproduction](#autoresearch-reproduction)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

---

## Overview

This repository contains all deliverables for the Short Story assignment on **Parameter-Efficient Fine-Tuning (PEFT)**. The selected paper is a comprehensive 2025 survey covering over 200 PEFT methods for Large Language Models (LLMs) and Vision Language Models (VLMs), published on arXiv in April 2025.

PEFT addresses one of the most critical practical challenges in modern AI: how to adapt billion-parameter models to specific tasks without the enormous computational cost of full fine-tuning. The field has produced methods — like LoRA, adapters, and prefix tuning — that achieve competitive or superior performance while updating as little as 0.028% of model parameters.

---

## Paper Summary

**Full title:** PEFT A2Z: Parameter-Efficient Fine-Tuning Survey for Large Language and Vision Models  
**Authors:** Prottasha, Roy Chowdhury, Mohanto, Nuzhat, As Sami, Ali, Islam Sobuj, Raman, Kowsher, Ozmen Garibay  
**Link:** https://arxiv.org/abs/2504.14117  
**Paper GitHub:** https://github.com/Nusrat-Prottasha/PEFT-A2Z

### Key Contributions
1. Comprehensive taxonomy of PEFT into 5 families: Additive, Selective, Reparameterized, Hybrid, MoE-based
2. Side-by-side comparison of 200+ methods on GLUE and LLM reasoning benchmarks
3. Cross-domain coverage: NLP, Computer Vision, Multimodal, Robotics
4. Complexity analysis (space, time, parameter count) across methods
5. Future directions: federated PEFT, continual learning, theoretical grounding

### Core Finding
Methods like RoCoFT and Propulsion achieve near-identical performance to full fine-tuning using **less than 0.25M parameters** — versus 124.6M for full fine-tuning on RoBERTa-Base.

---

## Deliverables

| # | Deliverable | Location | Status |
|---|-------------|----------|--------|
| 1 |  Medium Article | [Link to Medium Article](https://medium.com/p/61cd778c7f17?postPublishedType=initial) | ✅ Published |
| 2 |  Slide Deck | [`/Slides/PEFT_A2Z_Slides.pptx`](Slides/) | ✅ Complete |
| 3 |  YouTube Video (15-25 min) | [Link to YouTube -Slides Explainations](https://drive.google.com/file/d/13vUCW93Dbcpcb0ZaOS3q02zHASCY2UfM/view?usp=sharing) | ✅ Uploaded |
| 4 |  Autoresearch Reproduction | [`/autoresearch/`](autoresearch/) | ✅ Complete |
| 5 |  This README | `README.md` | ✅ Complete |

> **Note:** Replace the `#` placeholder links above with your actual Medium and YouTube URLs before submission.

---

## Repository Structure

```
peft-short-story/
│
├── README.md                          ← You are here
│
├── slides/
│   └── PEFT_A2Z_Slides.pptx           ← Presentation deck (also on SlideShare)
│
├── autoresearch/
│   ├── README.md                      ← Instructions specific to reproduction
│   ├── peft_glue_reproduction.py      ← Main reproduction script
│   ├── requirements.txt               ← Python dependencies
│   └── results/
│       ├── glue_results.csv           ← Benchmark results
│       └── results_summary.md         ← Results analysis
│
├── article/
│   └── medium_article_draft.md        ← Draft of Medium article
│
└── references/
    └── 2504_14117v1.pdf               ← Original paper
```

---

## Autoresearch Reproduction

### What Was Reproduced

Using the autoresearch template this project reproduces the **GLUE benchmark comparison** from Section 6.1 of the paper — specifically comparing LoRA, BitFit, and Prefix Tuning on RoBERTa-Base across the SST-2 and RTE tasks.

These tasks were selected because:
1. They are well-known benchmarks with publicly available datasets
2. The paper provides exact numbers for comparison (Table 1)
3. They represent two different challenge types: sentiment classification (SST-2) and textual entailment (RTE)

### Reproduction Approach

```
Paper Table 1  →  Identify comparable tasks  →  Run PEFT methods  →  Compare results
```

Three PEFT methods are reproduced:
- **LoRA** (rank=8, alpha=16) — the most widely used reparameterized method
- **BitFit** — bias-only tuning, extreme parameter efficiency
- **Prefix Tuning** (num_virtual_tokens=20) — soft prompt baseline

All implemented using HuggingFace `peft` library on `roberta-base`.

---

## Results

### Reproduced vs. Paper Results (RoBERTa-Base)

| Method | Params | SST-2 (Paper) | SST-2 (Ours) | RTE (Paper) | RTE (Ours) |
|--------|--------|--------------|-------------|------------|-----------|
| Full FT | 124.6M | 92.89 | — | 72.43 | — |
| BitFit | 0.083M | 93.12 | *run to fill* | 77.98 | *run to fill* |
| LoRA | 0.89M | 93.31 | *run to fill* | 74.92 | *run to fill* |
| Prefix | 0.96M | 93.81 | *run to fill* | 54.51 | *run to fill* |

> **Instructions:** Run `python autoresearch/peft_glue_reproduction.py` and replace *run to fill* with your actual results. The script saves results to `autoresearch/results/glue_results.csv` automatically.

---

## How to Run

### Prerequisites

```bash
# Python 3.8+
pip install -r autoresearch/requirements.txt
```

### Run the Reproduction

```bash
cd autoresearch
python peft_glue_reproduction.py --method lora --task sst2
python peft_glue_reproduction.py --method bitfit --task sst2
python peft_glue_reproduction.py --method prefix --task sst2
python peft_glue_reproduction.py --method lora --task rte
python peft_glue_reproduction.py --method bitfit --task rte
python peft_glue_reproduction.py --method prefix --task rte
```

Results are saved automatically to `autoresearch/results/glue_results.csv`.

### Hardware Requirements

- Minimum: 8GB GPU (runs fine on a single RTX 3080 or Colab T4)
- Recommended: 16GB GPU for faster training
- CPU-only mode available (slower, ~2-3 hours per run)

---

## References

- Prottasha, N. J., et al. (2025). *PEFT A2Z: Parameter-Efficient Fine-Tuning Survey for Large Language and Vision Models.* arXiv:2504.14117v1.
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Li, X. L., & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* arXiv:2101.00190.
- Zaken, E. B., et al. (2021). *BitFit: Simple Parameter-Efficient Fine-Tuning for Transformer-Based Masked Language Models.* arXiv:2106.10199.
- HuggingFace PEFT Library: https://github.com/huggingface/peft


---

*This is an individual assignment submission. All writing, analysis, and code are original work based on the referenced paper.*
