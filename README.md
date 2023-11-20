---
title: Code Generation With Language Specific LoRa Models
emoji: 🏆
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.21.0
app_file: app.py
pinned: false
license: openrail
---

## Demo:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://code-llms.streamlit.app/)


## **Dataset Preparation for Language Models on GitHub Code**


### **Dataset Source and Language Selection**

For this project, I primarily used TheStack Corpus, an open-source code dataset featuring a  3TB of GitHub data spread across a diverse range of 48 programming languages. To keep things focused, I narrowed down my scope to a subset of programming languages: Ruby, Swift, Rust, and Java. These languages were carefully chosen to represent the GitHub language distribution and the varying levels of resource availability.

### **Preprocessing Steps**

Here's a rundown of the steps I took to prepare the datasets:

1. **Language Choice**: I handpicked the languages mentioned earlier to ensure a well-rounded representation.
2. **Data Filtering**: To ensure quality, I filtered out files that didn't meet certain criteria, such as having an average line length exceeding 100 characters, a maximum line length exceeding 1000 characters, or an alphabet-to-numeric character ratio of less than 25%.
3. **Data Splitting**: After filtering, I divided the remaining files into train, validation, and test sets using a 90-5-5 ratio, respectively.

### **Tokenization Approach**

For tokenization, I employed Byte Pair Encoding (BPE) tokenizers, which are a proven technique in code processing. These tokenizers include special tab and white space tokens into the existing GPT-2 vocabulary \cite{gpt2}. To create training sequences, I tokenized the code in the files, and these tokens were then concatenated to form sequences of up to 2048 tokens. I also made use of special separator tokens to differentiate between different files. In some experiments, sequences of varying lengths were used.

## **Colab Notebooks**

These notebooks were used to train the models and generate the results. Copies of these notebooks can be found in the GitHub repository and in respective folders within the .zip file (named demo.ipynb). The notebooks are also linked below:

- [Full Fine-tuning Notebook](https://colab.research.google.com/drive/1BuRz-HBFCjxpmJfMg7QedbfNDXPl7Kap?usp=sharing)
- [LoRa Fine-tuning Notebook](https://colab.research.google.com/drive/1iWzsUeih_ObBJwmOkuD5D9Wm72eiRbQV?usp=sharing)
- [Perplexity Calculation Notebook](https://colab.research.google.com/drive/105aYjjovxfWKRifK5uzDfoQ2ZrTykoa4?usp=sharing)
- [Code Generation  Notebook](https://colab.research.google.com/drive/1gQ2GOwz40tNqF8UDakGsiZngJVt21DHI?usp=sharing)
- [Hyperparamter tuning Notebook](https://colab.research.google.com/drive/10ZIvvJml4cDMPVBPH_4QlLrU091XdGDA?usp=sharing)

## **Links to Trained Models, Datasets, and Inefernce Engine**

- [Java Model](https://huggingface.co/ammarnasr/codegen-350M-mono-java)
- [Java Dataset](https://huggingface.co/datasets/ammarnasr/the-stack-java-clean)

- [Ruby Model](https://huggingface.co/ammarnasr/codegen-350M-mono-ruby)
- [Ruby Dataset](https://huggingface.co/datasets/ammarnasr/the-stack-ruby-clean)

- [Rust Model](https://huggingface.co/ammarnasr/codegen-350M-mono-rust)
- [Rust Dataset](https://huggingface.co/datasets/ammarnasr/the-stack-rust-clean)

- [Swift Model](https://huggingface.co/ammarnasr/codegen-350M-mono-swift)
- [Swift Dataset](https://huggingface.co/datasets/ammarnasr/the-stack-swift-clean)

- [Inference Engine](https://huggingface.co/spaces/ammarnasr/Code-Generation-with-Language-Specific-LoRa-Models)












## Colab Notebooks
Code-LLM-finetuning-V7.ipynb: <a href="https://colab.research.google.com/drive/1BuRz-HBFCjxpmJfMg7QedbfNDXPl7Kap?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=10></a>  

Code-LLM-finetuning-LoRa-V7.ipynb <a href="https://colab.research.google.com/drive/1iWzsUeih_ObBJwmOkuD5D9Wm72eiRbQV?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=10></a>  


Code-LLM-Perplexity.ipynb <a href="https://colab.research.google.com/drive/105aYjjovxfWKRifK5uzDfoQ2ZrTykoa4?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=10></a>  


Code-LLM-Generation-Python.ipynb <a href="https://colab.research.google.com/drive/1gQ2GOwz40tNqF8UDakGsiZngJVt21DHI?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=10></a>  


Code-LLM-Generation-Java.ipynb <a href="https://colab.research.google.com/drive/13ocCjQwO0-hwkEt1xWNzRVFkfzMBn459?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=10></a>  

Code-LLM-Ablations.ipynb <a href="https://colab.research.google.com/drive/10ZIvvJml4cDMPVBPH_4QlLrU091XdGDA?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=10></a>  


Code-LLM-Checkpoint-Generations.ipynb: <a href="https://colab.research.google.com/drive/10J_6AVmGv5GKX1vJ-A-4SfKFpRr8Q3HZ?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=10></a> 





