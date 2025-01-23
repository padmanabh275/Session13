# SmolLM2: A Small Language Model Implementation

SmolLM2 is a lightweight transformer-based language model implementation using PyTorch Lightning. It features a modular architecture, configurable training, and easy deployment.

## Features
- Transformer-based architecture with configurable parameters
- Two-phase training strategy
- Mixed precision training support
- Gradient accumulation
- Automatic checkpoint management
- WandB integration for experiment tracking
- Gradio interface for model inference

## Installation

1. Clone the repository:
bash
git clone <repository-url>
cd smollm2

2. Create a virtual environment and install dependencies:

bash
conda create -n torch_env python=3.10
conda activate torch_env
Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
Install other requirements
pip install -r requirements.txt


## Model Architecture

The SmolLM2 model consists of:
- Token embeddings
- Rotary positional embeddings
- Multi-head attention with grouped-query attention
- RMSNorm for layer normalization
- MLP with SwiGLU activation
- Configurable model sizes

Key parameters:
- Hidden size: 256
- Attention heads: 4
- KV heads: 2
- Layers: 6
- Intermediate size: 512
- Max sequence length: 256

## Configuration

The model and training parameters can be configured through `config.yaml`:

can we use small dataset 

Yes, let's modify the configuration to use a smaller dataset for testing. We'll use a smaller subset of wikitext and reduce the model size. Here's the updated config.yaml:

Key changes:

Smaller model architecture:
Reduced hidden size
Fewer attention heads
Fewer layers
Smaller sequence length
Training settings:
Fewer training steps
More frequent sampling
Smaller learning rate
Smaller batch sizes

Dataset:

Added split_ratio to use only 1% of the data
Using wikitext-103 with a small fraction instead of full wikitext-2


## Training

Training Logs are available at:

https://wandb.ai/padmanabhbosamia/smol-lm2/runs/sj4i6232/logs

The training process consists of two phases:

1. Initial Training (5000 steps):


2. Fine-tuning (50 steps):
- Automatically starts after phase 1
- Uses the best checkpoint from phase 1

Training features:
- Automatic mixed precision
- Gradient clipping
- Learning rate scheduling
- Regular checkpointing
- Generation samples during training
- WandB logging

## Model Inference

Use the Gradio interface to interact with the trained model:

This will start a web interface where you can:
- Enter custom prompts
- Adjust generation parameters
- See model outputs in real-time

## Project Structure

- `model.py`: Model architecture implementation
- `config.py`: Configuration management
- `train_script.py`: Training loop implementation
- `app.py`: Gradio interface
- `config.yaml`: Model and training configuration
- `requirements.txt`: Project dependencies

## Training Data

The model is trained on the Wikitext-103 dataset, using:
- 1% of the dataset for faster training
- Dynamic batching
- Efficient data loading

## Hardware Requirements

Recommended specifications:
- CUDA-capable GPU (8GB+ VRAM)
- 16GB+ RAM
- Python 3.8+
- PyTorch 2.0+

## License

MIT

## Acknowledgments

- Based on transformer architecture
- Uses PyTorch Lightning for training
- Integrates Hugging Face transformers
