# Sentiment Analysis with LSTM (PyTorch)

This project performs binary sentiment classification on tweets using an LSTM-based neural network implemented in PyTorch.

## Objective

To classify tweets as either positive or negative using deep learning techniques and apply standard natural language processing (NLP) steps, including tokenization, vocabulary building, and sequence padding.

## Dataset

- Dataset: Sentiment140 (160,000 labeled tweets)
- Labels: `0` (Negative), `4` (Positive), mapped to `0` and `1`

## Preprocessing

- Lowercasing, removing URLs, mentions, and non-alphanumeric characters using regex
- Tokenization and optional stemming using `nltk.SnowballStemmer`
- Removing stopwords
- Vocabulary construction using `collections.Counter`
- Sequence padding and truncation to a fixed length (`max_len = 100`)
- Train/dev/test split using `train_test_split`

## Model Architecture

- Embedding Layer
- Bidirectional LSTM (`2` layers, `hidden_dim = 32`)
- Dropout (`p = 0.7`)
- Fully Connected Layer (output size = `1`)
- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW` with weight decay
- Batch size: `256`
- Device: CUDA (if available)

## Training Strategy

- Early stopping based on validation accuracy
- Track train/dev loss and accuracy across epochs
- Save best model based on validation performance

## Evaluation

- Final Test Accuracy: **78.43%**
- Classification Report:
  - Precision (class 1): 0.82
  - Recall (class 1): 0.72
  - F1-score (macro): 0.78

## Visualization

Accuracy and loss curves are plotted for both training and validation sets over all epochs.

## Purpose

This notebook was created for learning and practicing the full workflow of sentiment analysis with LSTM using PyTorch, from preprocessing raw text to evaluating model performance.
