# Off-Target Mitigation in NLLB

This repository provides scripts and models to evaluate and improve the performance of the [NLLB](https://ai.facebook.com/research/no-language-left-behind/) (No Language Left Behind) multilingual translation system using **Target Language Prediction (TLP)** to mitigate off-target translations. It includes training and evaluation pipelines for both baseline and fine-tuned models.

---

## Repository Structure

### 🔹 Translation Evaluation

| Script | Description |
|--------|-------------|
| `NLLB_Baseline_test_eng.py` | Evaluate baseline NLLB model on English → many translation. |
| `NLLB_Baseline_test_mal.py` | Evaluate baseline NLLB model on Malayalam → many translation. |
| `NLLB_Baseline_test_spa.py` | Evaluate baseline NLLB model on Spanish → many translation. |
| `NLLB_TLP_test_eng.py` | Evaluate fine-tuned TLP model on English → many. |
| `NLLB_TLP_test_mal.py` | Evaluate fine-tuned TLP model on Malayalam → many. |
| `NLLB_TLP_test_spa.py` | Evaluate fine-tuned TLP model on Spanish → many. |

### 🔹 Language Identification

| Script | Description |
|--------|-------------|
| `languageIdentifier_Train.py` | Trains a language identifier using FLORES-200 and XLM-RoBERTa. |
| `languageIdentifierEvaluation.py` | Evaluates the trained language identification model. |

### 🔹 Correlation Analysis

| Script | Description |
|--------|-------------|
| `correlationAnalysis.py` | Analyzes correlation between off-target rates and spBLEU. |
| `correlation_visualisation.py` | Visualizes off-target vs BLEU score trends. |

### 🔹 TLP Training

| Script | Description |
|--------|-------------|
| `NLLB_TLP_training.py` | Fine-tunes the NLLB model with Target Language Prediction (TLP). |

---

##  Usage

Run the provided `run.sh` script to install all required dependencies using Python 3.8.
