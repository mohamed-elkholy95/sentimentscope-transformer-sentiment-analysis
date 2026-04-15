<div align="center">

<img src="assets/sentimentscope-banner.jpg" alt="SentimentScope banner" width="100%" />

# SentimentScope

### Transformer-Powered Movie Review Sentiment Analysis

<p>
A sentiment analysis project built with PyTorch and Hugging Face tooling to classify IMDB movie reviews with a transformer-based architecture.
</p>

<p>
  <img src="https://img.shields.io/badge/Model-Transformer-f97316?style=for-the-badge" alt="Transformer Model" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-ef4444?style=for-the-badge" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Dataset-IMDB-6366f1?style=for-the-badge" alt="IMDB Dataset" />
</p>

</div>

---

## The Problem

Movie review sentiment classification is a simple problem to explain and a strong one for evaluating NLP execution. The goal is to classify IMDB reviews as `positive` or `negative` using a transformer-based architecture with a reproducible training and evaluation flow.

## Project Overview

- IMDB review preprocessing pipeline for binary classification
- `bert-base-uncased` tokenization with Hugging Face
- custom PyTorch transformer-based sentiment classifier
- training loop with validation tracking and checkpoint saving
- test evaluation workflow aligned with course success criteria

## Technical Highlights

**ML Engineering**
Data preparation, batching, modeling, training, and evaluation in one end-to-end workflow.

**NLP**
Tokenization, sequence modeling, and sentiment classification on a standard benchmark dataset.

**Frameworks**
PyTorch for model/training logic and Hugging Face for tokenizer integration.

**Engineering Quality**
Clear structure, readable notebook flow, and documentation that supports quick review.

## Architecture Flow

```text
IMDB Reviews
   -> Text preprocessing
   -> BERT tokenizer
   -> Token IDs + attention masks
   -> Transformer-based sentiment model
   -> Validation tracking
   -> Best checkpoint
   -> Final test evaluation
```

## Results

- Includes full validation and test evaluation flow in the notebook
- Built to satisfy the project target of exceeding `75%` test accuracy
- Saves the strongest validation checkpoint for final evaluation reuse

## Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</p>

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
`-- starter/
    |-- README.md
    `-- SentimentScope_starter.ipynb
```

## Run Locally

```bash
git clone https://github.com/mohamed-elkholy95/sentimentscope-transformer-sentiment-analysis.git
cd sentimentscope-transformer-sentiment-analysis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open `starter/SentimentScope_starter.ipynb` and run the notebook cells in order.

## Notes

- Originally developed as the capstone project for a Udacity nanodegree program
- Uses the IMDB sentiment dataset for binary classification
