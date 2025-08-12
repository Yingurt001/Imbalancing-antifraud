# Kolmogorov–Arnold Networks-based GRU and LSTM for Loan Default Early Prediction

## Overview
This repository contains the implementation and evaluation of **GRU-KAN** and **LSTM-KAN** models for **early loan default prediction** using time series anomaly detection.  
The models combine **Kolmogorov–Arnold Networks (KAN)** with **Gated Recurrent Units (GRU)** and **Long Short-Term Memory (LSTM)** to enhance predictive capability, especially in **out-of-time (OOT)** scenarios.

The objective is to predict loan defaults **months in advance**, enabling financial institutions to take preventive measures before risks materialize.

---

## Key Contributions
1. **KAN-enhanced GRU and LSTM**: Dynamic activation functions for flexible modeling of complex non-linear patterns in loan repayment behavior.
2. **Early prediction with blank intervals**: Simulating real-world scenarios where predictions are needed months ahead of defaults.
3. **Transformer comparison**: LSTM-Transformer is used as a baseline to examine self-attention's potential in loan default prediction.
4. **Robustness to OOT data**: Models perform strongly when trained and tested on data from different years, improving real-time applicability.

---

## Dataset
- **Source**: [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
- **Target Variable**: Default = 1 if loan delinquency ≥ 3 months, else 0.
- **Input Features**: Assistance status code, current actual UPB, current deferred UPB, current interest rate, estimated loan-to-value, interest-bearing UPB-delta.
- **Train/Test Split**: Train on 2019 Q1 data, test on 2020 Q1 (OOT setup).
- **Class Imbalance Handling**: Random undersampling.

---

## Methodology
### Model Architecture
- **Feature Extraction**: GRU or LSTM layers (128 + 64 units) with Batch Normalization.
- **Non-linear Modeling**: KAN layer replaces linear weights with spline-parameterized univariate functions.
- **Output Layer**: Fully connected + dropout + sigmoid for binary classification.

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score, AUC.

---

## Experimental Scenarios
1. **Feature Window Lengths** (12–27 months)  
   - Goal: Examine sensitivity to input history length.
2. **Early Prediction with Blank Intervals** (3–8 months gap)  
   - Goal: Assess performance predicting defaults months ahead.
3. **Sample Size Variations** (500k – 5M records)  
   - Goal: Test robustness with limited training data.
4. **Generalization Across Cohorts** (2018–2022)  
   - Goal: Evaluate adaptability under concept drift.

---

## Results Summary

### 1. Feature Window Lengths
- GRU-KAN and LSTM-KAN outperform baselines consistently.
- KAN layer narrows performance gap between GRU and LSTM.
- LSTM-Transformer shows high variability, excelling only in specific window lengths.

### 2. Early Prediction
- **GRU-KAN leads in Accuracy, Recall, and F1 across all intervals**.
- Models achieve same performance at **5-month gap** as baselines do at **3-month gap**, giving **2 months extra reaction time**.
- Accuracy drop (3 → 8 months):  
  - GRU-KAN: ~4.4%  
  - Baseline GRU: ~16.17%

### 3. Sample Size Sensitivity
- Proposed models outperform baselines, especially with smaller datasets.
- KAN improves GRU more significantly than LSTM.

### 4. Generalization Across Cohorts
- GRU-KAN shows highest accuracy and F1 in most cases.
- Concept drift reduces performance over time, but proposed models remain robust.

---

## Key Insights
- **KAN integration improves robustness**, especially in early prediction and small-data scenarios.
- **GRU-KAN consistently delivers best recall**, critical for minimizing missed defaults.
- Early prediction capability provides **practical risk management benefits** in finance.
- **OOT adaptability** makes models suitable for real-time deployment.

---

## Limitations & Future Work
- Only one dataset tested — need evaluation on multiple sources.
- Undersampling used; future work could explore SMOTE or GAN-based resampling.
- Adaptive learning for concept drift is a promising direction.

---

## Citation
If you use this work, please cite:
<pre> ```bibtex @article{yang2025kanloan, title={Kolmogorov–Arnold Networks-based GRU and LSTM for Loan Default Early Prediction}, author={Yue Yang and Zihan Su and Ying Zhang and Chang Chuan Goh and Yuxiang Lin and Anthony Graham Bellotti and Boon Giin Lee}, journal={arXiv preprint arXiv:2507.13685}, year={2025} } ``` </pre>
