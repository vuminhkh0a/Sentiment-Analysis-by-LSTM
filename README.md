# Sentiment Analysis with LSTM (PyTorch)

## Purpose

This is my first project to learn about NLP by using LSTM networks.


## Objective

To classify tweets as either positive or negative.

## Dataset

- Dataset: Sentiment140 (160,000 labeled tweets)
- Labels: `0` (Negative), `4` (Positive)
  
## Preprocessing

- Lowercasing, removing URLs, mentions, and non-alphanumeric characters using regex
- Tokenization and optional stemming using `nltk.SnowballStemmer`
- Removing stopwords
- Vocabulary construction using `collections.Counter`
- Sequence padding and truncation to a fixed length (`max_len = 100`)
- Train/dev/test split using `train_test_split`

## Model architecture

- Embedding Layer
- Bidirectional LSTM (`2` layers, `hidden_dim = 32`)
- Dropout (`p = 0.7`)
- Fully Connected Layer (output size = `1`)
- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW` with weight decay
- Batch size: `256`
- Device: CUDA (if available)

## Training

- Epochs: 20  
- Loss function: `nn.BCEWithLogitsLoss()`  
- Optimizer: `torch.optim.AdamW`  
  - Learning rate: `0.001`  
  - Weight decay: `0.07`  
- Batch size: 256  
- Early stopping with patience = 5  

## Evaluation

- Final Test Accuracy: **78.43%**
- Classification Report:
  - Precision (class 1): 0.82
  - Recall (class 1): 0.72
  - F1-score: 0.78


