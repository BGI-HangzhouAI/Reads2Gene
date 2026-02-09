# Reads2Gene: DNA Sequence Assembly via Reinforcement Learning

## 1. Introduction

Reads2Gene is a project that leverages reinforcement learning (RL) to train large language models (LLMs) for DNA sequence assembly tasks. The project uses the verl framework to fine-tune models like Qwen3-4B-Instruct for accurately assembling DNA reads into complete gene sequences.

The project addresses the challenge of reconstructing full DNA sequences from fragmented reads, which is a fundamental problem in genomics and bioinformatics. By applying RLHF (Reinforcement Learning from Human Feedback) techniques, the model learns to generate accurate and complete DNA sequences from input reads.

## 2. Model Information

### Base Model
- **Model**: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Framework**: verl (Volcano Engine Reinforcement Learning)
- **Training Method**: GRPO (Group Relative Policy Optimization) / PPO (Proximal Policy Optimization)

### Training Data

The training data consists of:
- **Dataset Generation**: Synthetic DNA sequences generated with various genomic features (intergenic regions, GC-rich regions, CpG islands, repeats, genes)
- **Data Format**: JSON files containing DNA sequences, questions, and ground truth answers
- **Data Processing**: Converted to parquet format for verl training pipeline
- **Sequence Length**: Variable lengths (typically 2000-3500 base pairs)

The dataset generation scripts are located in `train/dataset_generation/`:
- `generate_dataset.py`: Main dataset generator with complex sequence assembly
- `generate_dataset.py`: Base dataset generation utilities
- `generator_complex.py`: Complex chromosome assembly generator

### Reward Functions

The project implements multiple reward functions for RL training:

1. **Target Curve Reward** (`RL/reward_function_target_curve.py`): 
   - Calculates rewards based on sequence alignment and similarity metrics
   - Includes coverage score, format penalties, loop penalties, and edit distance penalties

2. **Equal Weight Reward** (`RL/reward_function_equal_weight.py`):
   - Balanced reward calculation with equal weighting of different metrics

3. **Batch Random Reward** (`RL/reward_function_batch_random.py`):
   - Random sampling-based reward calculation for batch processing

### Key Features

- **DNA Sequence Alignment**: Uses BioPython for pairwise sequence alignment
- **Similarity Scoring**: Multiple metrics including coverage, edit distance, and format validation
- **Noise Handling**: Supports training with noisy reads and echo sequences
- **Format Validation**: Ensures output follows XML format (`<answer>...</answer>`)

## 3. Performance Evaluation

Evaluation scripts are located in `evaluate/Script/`:

- **Scoring**: `evaluate/Script/data_process/calculate_score.py` - Calculates DNA sequence scores
- **DNA Scoring**: `evaluate/Script/data_process/dna_score.py` - Core DNA sequence evaluation metrics
- **Generation**: `evaluate/Script/generation/` - Model inference and generation scripts

Evaluation metrics include:
- Sequence similarity scores
- Coverage metrics
- Edit distance penalties
- Format validation scores

## 4. Quickstart

### Prerequisites

- Python 3.8+
- CUDA-capable GPU
- verl framework (included in `train/verl/`)
- BioPython for sequence alignment
- PyTorch, FSDP, vLLM for model training and inference

### Installation

1. Clone the repository:
```bash
cd /path/demo/Reads2Gene
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # If available
```

3. Set up verl framework (if not already configured):
```bash
cd train/verl
pip install -e .
```

### Dataset Preparation

1. Generate training dataset:
```bash
cd train/dataset_generation
python generate_dataset_20260107.py
```

2. Convert to parquet format for verl:
```bash
cd train/json2parquet
python prepare_data4verl_3500.py
```

### Training

Run RL training with GRPO:
```bash
cd RL
bash run_qwen3-4b_3500.sh
```

Key training parameters:
- Model: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- Training batch size: 16
- Max prompt length: 22000 tokens
- Max response length: 10000 tokens
- Learning rate: 1e-6
- Reward function: Custom reward function from `RL/reward_function_target_curve.py`

### Evaluation

1. Generate predictions:
```bash
cd evaluate/Script/generation
bash run_eval_by_api.sh
```

2. Calculate scores:
```bash
cd evaluate/Script/data_process
python calculate_score.py --input <input_file> --output <output_file>
```

## 5. Project Structure

```
Reads2Gene/
├── README.md                    # This file
├── train/                       # Training code
│   ├── dataset_generation/     # Dataset generation scripts
│   ├── json2parquet/           # Data format conversion
│   └── verl/                   # verl RL framework
├── RL/                         # Reinforcement learning scripts
│   ├── reward_function_*.py   # Reward function implementations
│   └── run_qwen3-4b_3500.sh   # Main training script
└── evaluate/                   # Evaluation code
    ├── Script/
    │   ├── data_process/       # Scoring and evaluation
    │   └── generation/         # Model inference
    └── score_result/           # Evaluation results
```

## 6. License

This repository and the associated model weights are released under the [Apache License 2.0](LICENSE), and follow the licensing terms of the base model [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Please note that if you use the Qwen3-4B-Instruct-2507 model, you must comply with its own license and usage requirements.

This project is primarily intended to support genomics research, providing researchers with advanced AI capabilities and tools for human genome analysis. It is not intended for any use that violates applicable laws or regulations, nor for any activities prohibited by the relevant license agreements.

## 7. Citation and Acknowledgements

- **verl Framework**: [verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl)
- **Base Model**: [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **BioPython**: For DNA sequence alignment and analysis

## 8. Notes

- The project uses verl's hybrid programming model for flexible RL algorithm implementation
- Training requires significant GPU resources (tested on 8x GPU nodes)
- The reward functions can be customized in `RL/reward_function_*.py` files

## 9. Contact
For project-related questions, please raise an [issue]() or contact the project maintainer at huyuebei24@mails.ucas.ac.cn.

For general inquiries, you are also welcome to contact us at huyuebei24@mails.ucas.ac.cn.
