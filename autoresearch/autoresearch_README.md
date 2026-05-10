# Autoresearch Reproduction — PEFT GLUE Benchmarks

## What This Reproduces

This folder reproduces **Table 1 (Section 6.1)** from Prottasha et al. (2025),
comparing LoRA, BitFit, and Prefix Tuning on RoBERTa-Base across two GLUE tasks:
- **SST-2** — binary sentiment classification
- **RTE** — recognizing textual entailment (2-class NLI)


## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and at least 8GB GPU (or runs on CPU, slower).

## Run All Experiments

```bash
# SST-2
python peft_glue_reproduction.py --method lora --task sst2
python peft_glue_reproduction.py --method bitfit --task sst2
python peft_glue_reproduction.py --method prefix --task sst2

# RTE
python peft_glue_reproduction.py --method lora --task rte
python peft_glue_reproduction.py --method bitfit --task rte
python peft_glue_reproduction.py --method prefix --task rte

# Print comparison table
python peft_glue_reproduction.py --method lora --task sst2 --summary
```

## Expected Results vs. Paper

| Method | Task | Paper | Expected Range |
|--------|------|-------|----------------|
| LoRA | SST-2 | 93.31 | 92.5 – 93.8 |
| BitFit | SST-2 | 93.12 | 92.0 – 93.5 |
| Prefix | SST-2 | 93.81 | 92.5 – 94.0 |
| LoRA | RTE | 74.92 | 73.0 – 76.5 |
| BitFit | RTE | 77.98 | 75.0 – 79.5 |
| Prefix | RTE | 54.51 | 52.0 – 57.0 |

Small differences from the paper are expected due to random seed variation and
minor differences in training hyperparameters. The relative ranking of methods
should match the paper.

## Output

Results are saved to `results/glue_results.csv` automatically after each run.
