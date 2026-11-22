# Task 1 — sentiment distribution: results

Systems: S0-vader, S1-tfidf-lr, S2-zeroshot, S3-bert-len256, S4-roberta-len256. Metacritic test: 24,257 reviews (30 games, 3 classes from user score). Steam: 10,000 reviews (50 games, binary from the recommend flag; 3-class predictions collapsed by p(pos) vs p(neg)).

## 1. Review level

| System | MC acc | MC macro-F1 | MC F1 pos | MC F1 neg | MC F1 neu | Steam acc | Steam macro-F1 | Steam F1 pos | Steam F1 neg |
|---|---|---|---|---|---|---|---|---|---|
| S0-vader | 0.709 | 0.484 | 0.823 | 0.588 | 0.041 | 0.802 | 0.661 | 0.880 | 0.443 |
| S1-tfidf-lr | 0.819 | 0.733 | 0.903 | 0.789 | 0.508 | 0.832 | 0.776 | 0.888 | 0.664 |
| S2-zeroshot | 0.787 | 0.587 | 0.892 | 0.749 | 0.120 | 0.843 | 0.783 | 0.897 | 0.668 |
| S3-bert-len256 | 0.862 | 0.783 | 0.934 | 0.841 | 0.576 | 0.885 | 0.835 | 0.926 | 0.744 |
| S4-roberta-len256 | 0.877 | 0.801 | 0.943 | 0.856 | 0.604 | 0.892 | 0.846 | 0.930 | 0.762 |

## 2. Distribution level (per game, classify-and-count; ACC = adjusted)

### Metacritic test (3 classes)

| System | MAE | TVD | TVD (ACC) | Dominant agree | Bias pos (pp) | Bias neg (pp) | Pearson pos | ACC cond. |
|---|---|---|---|---|---|---|---|---|
| S0-vader | 0.120 | 0.181 | 0.517 | 0.73 | +14.1 | -2.5 | 0.826 | 6532.7 |
| S1-tfidf-lr | 0.038 | 0.057 | 0.081 | 0.97 | -4.8 | +1.1 | 0.987 | 2.2 |
| S2-zeroshot | 0.073 | 0.109 | 0.275 | 1.00 | +0.9 | +8.2 | 0.987 | 34.2 |
| S3-bert-len256 | 0.023 | 0.035 | 0.052 | 1.00 | -2.3 | -0.2 | 0.995 | 1.9 |
| S4-roberta-len256 | 0.019 | 0.029 | 0.046 | 0.97 | -1.7 | +0.1 | 0.997 | 1.8 |

### Steam (binary)

| System | MAE | TVD | TVD (ACC) | Dominant agree | Bias pos (pp) | Bias neg (pp) | Pearson pos | ACC cond. |
|---|---|---|---|---|---|---|---|---|
| S0-vader | 0.087 | 0.087 | 0.085 | 0.98 | +2.6 | -2.7 | 0.701 | 2.2 |
| S1-tfidf-lr | 0.118 | 0.118 | 0.093 | 0.90 | -11.8 | +11.8 | 0.836 | 1.1 |
| S2-zeroshot | 0.094 | 0.094 | 0.061 | 0.96 | -9.2 | +9.2 | 0.877 | 1.2 |
| S3-bert-len256 | 0.071 | 0.071 | 0.063 | 0.96 | -7.0 | +7.0 | 0.905 | 1.1 |
| S4-roberta-len256 | 0.072 | 0.072 | 0.062 | 0.96 | -7.2 | +7.2 | 0.926 | 1.1 |

Bias = mean(predicted share − gold share) in percentage points. ACC cond. = condition number of the confusion matrix ACC inverts (large = unstable correction).

## 3. Batch-size ablation (mean TVD vs the game's full gold distribution)

### Metacritic test

| System | n=10 | n=50 | n=100 | n=200 |
|---|---|---|---|---|
| S0-vader | 0.221 | 0.190 | 0.185 | 0.182 |
| S1-tfidf-lr | 0.160 | 0.085 | 0.072 | 0.064 |
| S2-zeroshot | 0.177 | 0.123 | 0.115 | 0.111 |
| S3-bert-len256 | 0.153 | 0.074 | 0.056 | 0.044 |
| S4-roberta-len256 | 0.149 | 0.070 | 0.052 | 0.039 |
| _sampling noise only_ | 0.146 | 0.063 | 0.042 | 0.026 |

### Steam

| System | n=10 | n=50 | n=100 | n=200 |
|---|---|---|---|---|
| S0-vader | 0.122 | 0.092 | 0.088 | 0.087 |
| S1-tfidf-lr | 0.154 | 0.124 | 0.120 | 0.118 |
| S2-zeroshot | 0.137 | 0.102 | 0.096 | 0.094 |
| S3-bert-len256 | 0.124 | 0.082 | 0.075 | 0.071 |
| S4-roberta-len256 | 0.123 | 0.082 | 0.075 | 0.072 |
| _sampling noise only_ | 0.088 | 0.034 | 0.020 | 0.000 |

The last row is the TVD of the batch's own gold labels: the floor any classifier faces from sampling alone.

## 4. Error analysis (S4-roberta-len256)

### Confusion — Metacritic test (3 classes)

| gold \ pred | positive | negative | neutral |
|---|---|---|---|
| positive | 14323 | 232 | 857 |
| negative | 172 | 4970 | 670 |
| neutral | 461 | 600 | 1972 |

### Confusion — Steam (3-class prediction vs binary gold)

| gold \ pred | positive | negative | neutral |
|---|---|---|---|
| positive | 6340 | 636 | 1119 |
| negative | 115 | 1359 | 431 |

### Steam: where the binary errors come from

| gold | 3-class pred | n | share of gold | binary correct |
|---|---|---|---|---|
| positive | positive | 6340 | 0.783 | 1.000 |
| positive | negative | 636 | 0.079 | 0.000 |
| positive | neutral | 1119 | 0.138 | 0.762 |
| negative | positive | 115 | 0.060 | 0.000 |
| negative | negative | 1359 | 0.713 | 1.000 |
| negative | neutral | 431 | 0.226 | 0.852 |

11.2% of all Steam reviews are recommended but predicted neutral; the binary collapse then decides them by p(pos) vs p(neg) alone.

### Metacritic: accuracy by user score (label boundaries at 4|5 and 7|8)

| score | label | n | accuracy | pred_neutral |
|---|---|---|---|---|
| 0.0 | negative | 2922 | 0.917 | 0.042 |
| 1.0 | negative | 884 | 0.885 | 0.096 |
| 2.0 | negative | 603 | 0.841 | 0.149 |
| 3.0 | negative | 657 | 0.782 | 0.202 |
| 4.0 | negative | 746 | 0.653 | 0.319 |
| 5.0 | neutral | 897 | 0.555 | 0.555 |
| 6.0 | neutral | 905 | 0.712 | 0.712 |
| 7.0 | neutral | 1231 | 0.674 | 0.674 |
| 8.0 | positive | 2332 | 0.741 | 0.236 |
| 9.0 | positive | 3570 | 0.935 | 0.054 |
| 10.0 | positive | 9510 | 0.974 | 0.012 |

### Accuracy by review length (words; the model sees at most 256 tokens)

Metacritic:

| words | n | accuracy |
|---|---|---|
| <25 | 5260 | 0.896 |
| 25-49 | 6325 | 0.893 |
| 50-99 | 5785 | 0.877 |
| 100-199 | 3883 | 0.871 |
| 200-399 | 1977 | 0.830 |
| 400+ | 1027 | 0.784 |

Steam (binary):

| words | n | accuracy |
|---|---|---|
| <25 | 4441 | 0.863 |
| 25-49 | 2046 | 0.909 |
| 50-99 | 1710 | 0.923 |
| 100-199 | 1069 | 0.930 |
| 200-399 | 516 | 0.893 |
| 400+ | 218 | 0.885 |

### Steam games with the largest error in % positive

| app_id | game | gold % pos | pred % pos | error (pp) |
|---|---|---|---|---|
| 346900 | AdVenture Capitalist | 92.5 | 66.5 | -26.0 |
| 340 | Half-Life 2: Lost Coast | 83.0 | 61.0 | -22.0 |
| 282800 | 100% Orange Juice | 91.0 | 70.0 | -21.0 |
| 420110 | NEKOPARA Vol. 2 | 98.5 | 80.5 | -18.0 |
| 345650 | Without Within | 91.5 | 77.0 | -14.5 |
| 339610 | Freestyle 2: Street Basketball | 68.5 | 54.0 | -14.5 |
| 341940 | Hatred | 77.5 | 64.5 | -13.0 |
| 218620 | PAYDAY 2 | 68.0 | 56.0 | -12.0 |

### Agreement between systems on Steam (binary labels)

|  | S0-vader | S1-tfidf-lr | S2-zeroshot | S3-bert-len256 | S4-roberta-len256 |
|---|---|---|---|---|---|
| S0-vader | 1.00 | 0.75 | 0.80 | 0.78 | 0.78 |
| S1-tfidf-lr | 0.75 | 1.00 | 0.84 | 0.87 | 0.87 |
| S2-zeroshot | 0.80 | 0.84 | 1.00 | 0.87 | 0.87 |
| S3-bert-len256 | 0.78 | 0.87 | 0.87 | 1.00 | 0.93 |
| S4-roberta-len256 | 0.78 | 0.87 | 0.87 | 0.93 | 1.00 |

### Label-noise check: 100 random Steam reviews labelled by reading the text (blind to the recommend flag)

Recommend flag (rows) vs text label (columns):

| flag \ text | positive | negative | neutral |
|---|---|---|---|
| positive | 66 | 2 | 11 |
| negative | 0 | 19 | 2 |

- text is neutral/mixed or has no opinion: 13% (of these, 85% are flagged as recommended)

- flag agrees with the text on polar reviews: 97.7% (87 reviews); disagreements: sarcasm, price complaint

- model binary label vs text (polar reviews): 96.6%; model binary vs flag (all): 93.0%; model 3-class vs text (all): 85.0%

Text label (rows) vs model 3-class prediction (columns):

| text \ model | positive | negative | neutral |
|---|---|---|---|
| positive | 61 | 1 | 4 |
| negative | 1 | 18 | 2 |
| neutral | 4 | 3 | 6 |

Qualitative samples of each error type: `error_samples.csv` (same folder).
