# RLReviewer Reproduction Guide

This directory contains the training and evaluation code for RLReviewer.

This reproduction package uses **LLaMA-Factory 0.9.4.dev0** and **verl 0.7.0.dev**.
LLaMA-Factory requires Python >= 3.9, while verl requires Python >= 3.10; therefore, all commands below use Python 3.10.
Please install the source code included in this package to preserve the versions and interfaces used in the experiments.

We recommend creating separate environments for SFT/inference and RL because packages such as PyTorch, vLLM, and FlashAttention have strict compatibility requirements.

## 1. Data Preparation

Download the released dataset from the [RLReviewer data archive](https://drive.google.com/file/d/1ThqWuZIyabiRlmnjPsNFBIBoQNbqBFJD/view?usp=sharing).
The additional [unseen dataset](https://drive.google.com/file/d/1s0Co3mi4cy75lx4Up-RSWEyOAV78WlbW/view?usp=drive_link) is provided for evaluating generalization on data not included in model training.
LLaMA-Factory uses JSON files for SFT and inference, while verl uses Parquet files for RL training.

Organize the data and models as shown below. The environment variables described in later sections can also be used to override these default paths.

```text
RLReviewer/
├── code/
│   ├── LLaMA-Factory-main/
│   │   └── data/
│   │       ├── cr_sft_train.json
│   │       ├── cr_sft_test.json
│   │       └── dataset_info.json
│   ├── verl-main/
│   └── evaluation/
├── data/
│   ├── cr_rl_train.parquet
│   └── cr_rl_val.parquet
├── models/
│   ├── Qwen2.5-Coder-7B-Instruct/
│   ├── all-MiniLM-L6-v2/
│   └── sft_model/
└── checkpoints/
```

LLaMA-Factory resolves the `cr_sft_train` and `cr_sft_test` dataset identifiers through `code/LLaMA-Factory-main/data/dataset_info.json`. These entries are already registered in the bundled configuration for Alpaca-format files containing `instruction`, `output`, and `system` fields:

```json
"cr_sft_train": {
  "file_name": "cr_sft_train.json",
  "columns": {
    "prompt": "instruction",
    "response": "output",
    "system": "system"
  }
},
"cr_sft_test": {
  "file_name": "cr_sft_test.json",
  "columns": {
    "prompt": "instruction",
    "response": "output",
    "system": "system"
  }
}
```

Place the downloaded files at the paths shown above. If their filenames or fields are changed locally, update the mappings accordingly. For details, see the bundled [LLaMA-Factory data format documentation](LLaMA-Factory-main/data/README.md) and the version-matched [official custom dataset documentation](https://github.com/hiyouga/LlamaFactory/blob/2a822178dea4d1c05f595521dd883a8e4f4e2e77/data/README.md).

Each verl Parquet record must contain the fields required by the trainer and reward function. In particular, `prompt` stores the conversational prompt, and `reward_model.ground_truth` must be a JSON string containing `line_number`, `rule`, and `comment`. A conceptual example is shown below:

```json
{
  "data_source": "rlreviewer",
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {
    "ground_truth": "{\"line_number\": 12, \"rule\": \"...\", \"comment\": \"...\"}"
  },
  "extra_info": {}
}
```

For details about the data structure required by verl, see the bundled [verl data preparation documentation](verl-main/docs/preparation/prepare_data.rst).

## 2. SFT

SFT uses **LLaMA-Factory 0.9.4.dev0**. The corresponding official instructions are available in the [LLaMA-Factory 0.9.4.dev0 installation section](https://github.com/hiyouga/LlamaFactory/blob/2a822178dea4d1c05f595521dd883a8e4f4e2e77/README.md#installation), and a copy of the [same version's documentation](LLaMA-Factory-main/README.md#installation) is included in this package. Following the official instructions for this version, install the bundled source code together with DeepSpeed and bitsandbytes for `0-sft.sh`, and vLLM for `1-infer.sh`:

```bash
conda create -n rlreviewer-sft python=3.10 -y
conda activate rlreviewer-sft

cd code/LLaMA-Factory-main
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[torch,metrics,deepspeed,bitsandbytes,vllm]" --no-build-isolation

llamafactory-cli version
cd ../..
```

The SFT script performs 4-bit LoRA training with DeepSpeed ZeRO-3. Set the W&B project name and API key. If the base model is not stored in the default directory, also set `MODEL_PATH`. Create the log directory and launch training from the LLaMA-Factory directory:

```bash
conda activate rlreviewer-sft
cd code/LLaMA-Factory-main

mkdir -p logs
export WANDB_API_KEY="<your-wandb-api-key>"
export WANDB_PROJECT="rlreviewer"
export MODEL_PATH="$(cd ../.. && pwd)/models/Qwen2.5-Coder-7B-Instruct"

bash 0-sft.sh
```

Training checkpoints are saved under `code/LLaMA-Factory-main/saves/${WANDB_PROJECT}/`. To merge an SFT LoRA adapter, first create a copy of `2-merge_lora.yaml`, set `model_name_or_path`, `adapter_name_or_path`, and `export_dir` in the copied file, and then export the model:

```bash
cd code/LLaMA-Factory-main
cp 2-merge_lora.yaml 2-merge_lora.local.yaml
# Modify only the three paths in 2-merge_lora.local.yaml.
llamafactory-cli export 2-merge_lora.local.yaml
```

When using the default RL directory layout in this guide, export the merged model to `RLReviewer/models/sft_model`.

## 3. RL

RL uses **verl 0.7.0.dev**, with FSDP as the training backend and vLLM as the inference backend. The source code under `verl-main/` matches the official `0.7.0.dev` version commit. Refer to the version-matched [official installation documentation](https://github.com/verl-project/verl/blob/55f651c94de75a1efcde8e14d079a0e70841a516/docs/start/install.rst) and [official vLLM 0.8 installation documentation](https://github.com/verl-project/verl/blob/55f651c94de75a1efcde8e14d079a0e70841a516/docs/README_vllm0.8.md). The same files are also included in this package as [`docs/start/install.rst`](verl-main/docs/start/install.rst) and [`docs/README_vllm0.8.md`](verl-main/docs/README_vllm0.8.md).

The official instructions for this version recommend Python 3.10 and the dependency installation script provided in the repository. This experiment uses only FSDP and vLLM, so the installation of SGLang and Megatron can be disabled:

```bash
conda create -n rlreviewer-rl python=3.10 -y
conda activate rlreviewer-rl

cd code/verl-main
python -m pip install --upgrade pip setuptools wheel
USE_SGLANG=0 USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install --no-deps -e .
python -m pip install nltk sentence-transformers
python -m nltk.downloader punkt punkt_tab

python -c "import verl, vllm; print('verl:', verl.__version__, 'vLLM:', vllm.__version__)"
cd ../..
```

The dependency installation script uses the package combination associated with the bundled verl source, including PyTorch 2.6.0, vLLM 0.8.5.post1, FlashAttention 2.7.4.post1, and FlashInfer 0.2.2.post1. The expected environment is Linux x86_64 with Python 3.10 and CUDA 12.x. If the host environment is incompatible with these binary packages, follow the version-matched official documentation above to use a Docker environment.

Before training, prepare the SentenceTransformer model used by `code/verl-main/reward_function.py`. The default is `sentence-transformers/all-MiniLM-L6-v2`; set `SENTENCE_TRANSFORMER_MODEL` when using a local copy. Then configure the merged SFT model, RL datasets, checkpoint output directory, and W&B credentials:

```bash
conda activate rlreviewer-rl
cd code/verl-main

export WANDB_API_KEY="<your-wandb-api-key>"
export MODEL_PATH="$(cd ../.. && pwd)/models/sft_model"
export CR_RL_TRAIN_PATH="$(cd ../.. && pwd)/data/cr_rl_train.parquet"
export CR_RL_VAL_PATH="$(cd ../.. && pwd)/data/cr_rl_val.parquet"
export CKPTS_DIR="$(cd ../.. && pwd)/checkpoints/rlreviewer"
export SENTENCE_TRANSFORMER_MODEL="$(cd ../.. && pwd)/models/all-MiniLM-L6-v2"

bash 0-rl.sh
```

`0-rl.sh` is configured for a single node with eight GPUs. Training checkpoints are written to `CKPTS_DIR`, and the training log is written to the current directory. The script uses the custom reward entry point `bleu_cosine_reward_fn` from `reward_function.py`.

The default comment-quality reward assigns equal weight to BLEU and SentenceBERT. To reproduce the single-signal ablations, set `COMMENT_BLEU_WEIGHT=1 COMMENT_SEMANTIC_WEIGHT=0` for BLEU only, or `COMMENT_BLEU_WEIGHT=0 COMMENT_SEMANTIC_WEIGHT=1` for SentenceBERT only.

## 4. Evaluation

Use the bundled vLLM inference script to generate predictions. `SFT_MODEL_PATH` should point to the merged SFT model, while `RL_CHECKPOINT_DIR` should point to the checkpoint directory containing the RL LoRA adapter. Before running inference, verify that the `adapter_name_or_path` assembled in `1-infer.sh` resolves to the existing `actor/lora_adapter` directory of the checkpoint to be evaluated.

```bash
conda activate rlreviewer-sft
cd code/LLaMA-Factory-main

export SFT_MODEL_PATH="$(cd ../.. && pwd)/models/sft_model"
export RL_CHECKPOINT_DIR="$(cd ../.. && pwd)/checkpoints/rlreviewer"

bash 1-infer.sh
```

The inference output is a JSONL file in which each record contains `predict` and `label`. Both fields must be JSON objects, or strings containing JSON objects, with the following structure:

```json
{
  "line_number": 12,
  "rule": "rule identifier",
  "comment": "code review comment"
}
```

Install the following two evaluation dependencies in the environment used to run the metric script:

```bash
python -m pip install nltk sentence-transformers
```

Then pass exactly one prediction JSONL path to the evaluator:

```bash
python code/evaluation/metric.py /path/to/predictions.jsonl
```

The evaluator prints four dataset-level average metrics to the terminal: **Location Accuracy**, **Rule Accuracy**, **BLEU**, and **SentenceBERT**. By default, it loads `sentence-transformers/all-MiniLM-L6-v2`. For offline evaluation or use of a local model, specify the model path through an environment variable without modifying the code:

```bash
SENTENCE_TRANSFORMER_MODEL=/path/to/all-MiniLM-L6-v2 \
python code/evaluation/metric.py /path/to/predictions.jsonl
```
