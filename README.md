# Replication Package

This repository provides the code and instructions needed to replicate the experiments from **“RLReviewer: Structured Code Review Comment Generation with Verification-Guided Reinforcement Learning.”**

## Code

All experiment code is located in the `code/` directory. The main scripts are:

- `sft.sh`: Runs supervised fine-tuning (SFT) of the base model on the code review dataset.
- `rl.sh`: Runs reinforcement learning (RL) fine-tuning using the verification-guided reward function.
- `reward_function.py`: Implements the reward function used during RL fine-tuning.

We use different frameworks for each stage:

- **SFT** is conducted with [LlamaFactory](https://github.com/hiyouga/LlamaFactory).
- **RL** is conducted with [verl](https://github.com/verl-project/verl).


## Data

The dataset can be downloaded from the following link: [data](https://drive.google.com/file/d/1ThqWuZIyabiRlmnjPsNFBIBoQNbqBFJD/view?usp=sharing)

- **JSON** files are used for SFT.
- **Parquet** files are used for RL fine-tuning.


## Acknowledgements

We sincerely thank the developers of the following open-source projects, whose contributions made this replication package possible:

- LlamaFactory: https://github.com/hiyouga/LlamaFactory  
- verl: https://github.com/verl-project/verl  
- vLLM: https://github.com/vllm-project/vllm