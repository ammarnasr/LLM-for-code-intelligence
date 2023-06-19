# Progress, Challenges and Next Steps

## Challenges

## Progress & Next Steps

## Relevant Literature:
### 1. Code Language Models:
#### i. Code Language Models Trained on Code Corpora only:
- [ ] [X](https://arxiv.org/pdf/2002.08155.pdf); Trained on one programming language (Python).
- [ ] [Y](https://arxiv.org/pdf/2007.15651.pdf); Trained on multiple programming languages.
#### ii. Code Language Models Trained on Code and Natural Language Corpora:
- [ ] [X](https://arxiv.org/pdf/2102.00821.pdf); Trained on one programming language (Python).
- [ ] [Y](https://arxiv.org/pdf/2106.09647.pdf); Trained on multiple programming languages.
### 2. Training Datasets:
#### i. Task Specific Datasets:
- [ ] [X](https://arxiv.org/pdf/2102.00821.pdf); Proplem Solving Tasks.
#### ii. General Datasets:
- [ ] [X](https://arxiv.org/pdf/2102.00821.pdf); CodeSearchNet Corpus.
### 3. Fine Tuning:
#### i. Full fine tuning:
- [ ] [X](https://arxiv.org/pdf/2102.00821.pdf); Full fine tuning on task specific dataset.
#### ii. Parameter efficient fine tuning:
- [ ] [X](https://arxiv.org/pdf/2102.00821.pdf); Parameter efficient fine tuning on task specific dataset.
### 4. Evaluation:
#### i. Evaluation Metrics:
- [ ] [X](https://arxiv.org/pdf/2102.00821.pdf); Similarity Metrics.
- [ ] [X](https://arxiv.org/pdf/2102.00821.pdf); Functional Metrics.
### 5. Further Reading:
#### i. Monolingual to Multilingual Transfer Learning:
- [ ] [X](https://arxiv.org/pdf/2106.09647.pdf); Monolingual to Multilingual Distillation.
#### ii. Qualtitative Analysis of Code Language Models:
- [ ] [X](https://arxiv.org/pdf/2106.09647.pdf); Qualtitative Analysis of Code Language Models.
#### iii. Code Language Models for Code Generation:
- [ ] [X](https://arxiv.org/pdf/2106.09647.pdf); Code Language Models for Code Generation.
#### iv. Infilling Sampling Strategies:
- [ ] [X](https://arxiv.org/pdf/2106.09647.pdf); Infilling Sampling Strategies.

## Experiments:
### Experiment 1: Using CodeGen 350M Mono-lingual Model for Code Generation
#### i. Baseline:
- [ ] A.Results of Pretrained Model on HumanEval Python.
- [ ] B.Results of Pretrained Model on HumanEval Java.
- [ ] C.Perplexity of Pretrained Model on GitHubCode Corpus, python subset.
- [ ] D.Perplexity of Pretrained Model on GitHubCode Corpus, java subset.
#### ii. Full Fine Tuning:
- [ ] A.Results of Full Fine Tuning on HumanEval Python.
- [ ] B.Results of Full Fine Tuning on HumanEval Java.
- [ ] C.Perplexity of Full Fine Tuning on GitHubCode Corpus, python subset.
- [ ] D.Perplexity of Full Fine Tuning on GitHubCode Corpus, java subset.
#### iii. Parameter Efficient Fine Tuning:
- [ ] A.Results of Parameter Efficient Fine Tuning on HumanEval Python.
- [ ] B.Results of Parameter Efficient Fine Tuning on HumanEval Java.
- [ ] C.Perplexity of Parameter Efficient Fine Tuning on GitHubCode Corpus, python subset.
- [ ] D.Perplexity of Parameter Efficient Fine Tuning on GitHubCode Corpus, java subset.
#### iv. Knowledge Distillation from CodeGen 1B Multilingual Model to CodeGen 350M Mono-lingual Model:
- [ ] A.Results of Knowledge Distillation on HumanEval Python.
- [ ] B.Results of Knowledge Distillation on HumanEval Java.
- [ ] C.Perplexity of Knowledge Distillation on GitHubCode Corpus, python subset.
- [ ] D.Perplexity of Knowledge Distillation on GitHubCode Corpus, java subset.
### Experiment 2: TBD
