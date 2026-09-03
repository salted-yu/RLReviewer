# RLReviewer Replication Package

This repository contains the replication package for **“RLReviewer: Structured Code Review Comment Generation with Verification-Guided Reinforcement Learning.”** It provides the training, inference, and evaluation code used to reproduce the RLReviewer workflow, together with the accompanying supplementary material.

RLReviewer generates a structured code review result with three fields:

```json
{
  "line_number": 12,
  "rule": "rule identifier",
  "comment": "code review comment"
}
```

The framework first learns code review generation through supervised fine-tuning (SFT), then improves the model through reinforcement learning (RL). The verification-guided reward evaluates output format, defect localization, rule identification, and review-comment quality.

## Workflow

| Stage | Purpose | Framework or entry point |
| --- | --- | --- |
| Data preparation | Prepare JSON data for SFT and Parquet data for RL | [Dataset archive](https://drive.google.com/file/d/1ThqWuZIyabiRlmnjPsNFBIBoQNbqBFJD/view?usp=sharing) |
| Supervised fine-tuning | Train a 4-bit LoRA model on code review examples | [LLaMA-Factory 0.9.4.dev0](code/LLaMA-Factory-main/), [`0-sft.sh`](code/LLaMA-Factory-main/0-sft.sh) |
| Reinforcement learning | Optimize the SFT model with the verification-guided reward | [verl 0.7.0.dev](code/verl-main/), [`0-rl.sh`](code/verl-main/0-rl.sh) |
| Inference | Generate structured review predictions with vLLM | [`1-infer.sh`](code/LLaMA-Factory-main/1-infer.sh) |
| Evaluation | Compute Location Accuracy, Rule Accuracy, BLEU, and SentenceBERT | [`metric.py`](code/evaluation/metric.py) |

## Repository Structure

```text
RLReviewer/
├── code/
│   ├── LLaMA-Factory-main/       # SFT, model export, and inference
│   │   ├── 0-sft.sh
│   │   ├── 1-infer.sh
│   │   └── 2-merge_lora.yaml
│   ├── verl-main/                # RL training and reward implementation
│   │   ├── 0-rl.sh
│   │   └── reward_function.py
│   ├── evaluation/
│   │   └── metric.py             # Four-metric evaluation CLI
│   └── README.md                 # Detailed reproduction instructions
├── output/
│   └── pdf/
│       └── supplementary_material.pdf
├── supplement/
│   ├── supplementary_material.tex
│   ├── robustness_to_ground_truth_noise.tex
│   ├── table_human_verified_overall.tex
│   ├── table_human_verified_by_rule_dimension.tex
│   └── table_reward_function_ablation.tex
└── README.md
```

Model weights, training checkpoints, and datasets are not bundled with this repository. Their paths are configured through environment variables documented in the reproduction guide.

## Data

Download the released dataset from the [RLReviewer data archive](https://drive.google.com/file/d/1ThqWuZIyabiRlmnjPsNFBIBoQNbqBFJD/view?usp=sharing).

The additional [unseen dataset](https://drive.google.com/file/d/1s0Co3mi4cy75lx4Up-RSWEyOAV78WlbW/view?usp=drive_link) is provided for evaluating generalization on data not included in model training.

- JSON files are used by LLaMA-Factory for supervised fine-tuning and inference.
- Parquet files are used by verl for reinforcement learning.
- Prediction files are stored as JSONL and evaluated with `code/evaluation/metric.py`.

The repository-level `.gitignore` excludes datasets and common archive formats so that local experimental data are not accidentally committed.

## Reproducing the Experiments

The complete installation and execution instructions are provided in the [code reproduction guide](code/README.md). That guide documents the bundled framework versions, recommended Python environments, expected directory layout, dataset formats, model paths, and commands for all four stages.

The main sequence is:

1. Download and arrange the dataset and base models.
2. Run SFT with `code/LLaMA-Factory-main/0-sft.sh`.
3. Merge the SFT LoRA adapter with `code/LLaMA-Factory-main/2-merge_lora.yaml`.
4. Run RL training with `code/verl-main/0-rl.sh`.
5. Generate predictions with `code/LLaMA-Factory-main/1-infer.sh`.
6. Evaluate the resulting JSONL file:

```bash
python code/evaluation/metric.py /path/to/predictions.jsonl
```

The SFT and RL environments are intentionally documented separately because their PyTorch, vLLM, and FlashAttention dependencies have strict compatibility requirements. The supplied RL configuration targets one node with eight GPUs.

## Supplementary Material

The supplementary material reports the robustness analysis on 351 human-verified test cases. It includes the overall comparison with baselines, results by rule dimension, and the reward-function ablation.

**[View the compiled supplementary material (PDF)](output/pdf/supplementary_material.pdf)**

The main LaTeX entry point is [`supplementary_material.tex`](supplement/supplementary_material.tex), and the editable content is split into semantically named source files under [`supplement/`](supplement/). To rebuild the PDF from the repository root, run:

```bash
mkdir -p output/pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf supplement/supplementary_material.tex
latexmk -c -outdir=output/pdf supplement/supplementary_material.tex
```

## Acknowledgements

We sincerely thank the developers of the open-source projects that support this replication package:

- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)
- [verl](https://github.com/verl-project/verl)
- [vLLM](https://github.com/vllm-project/vllm)
