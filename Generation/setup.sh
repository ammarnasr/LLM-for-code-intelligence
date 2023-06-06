#!/bin/bash

# Install requirements
pip install -r requirements.txt

# Login to Weights & Biases
wandb login

# Login to HuggingFace
huggingface-cli login
