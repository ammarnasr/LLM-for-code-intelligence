#!/bin/bash

# Install requirements
pip install -q -r requirements.txt

# Login to Weights & Biases
wandb login

# Login to HuggingFace
huggingface-cli login
