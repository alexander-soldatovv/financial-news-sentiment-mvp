# FinBERT vs Baseline

Both models are evaluated on the same stratified 20% test split with random state 42.

| Model | Macro precision | Macro recall | Inference ms/item |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8400 | 0.7862 | 0.040 |
| FinBERT | 0.9147 | 0.9244 | 9.742 |

| Class | Baseline precision | FinBERT precision | Baseline recall | FinBERT recall |
|---|---:|---:|---:|---:|
| Negative | 0.8529 | 0.9048 | 0.6905 | 0.9048 |
| Neutral | 0.8731 | 0.9664 | 0.9322 | 0.9416 |
| Positive | 0.7939 | 0.8730 | 0.7360 | 0.9270 |

FinBERT improves macro precision by 7.5 percentage points and macro recall by 13.8 percentage points. It is about 242x slower per item on the training machine, but its 9.7 ms latency remains practical for batch news screening.

FinBERT base checkpoint: `yiyanghkust/finbert-pretrain`.
