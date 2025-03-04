# MachineLearning_Labyrinth

# 1. Predicting Trending News on Sina Weibo

## Overview
This project utilizes machine learning techniques to predict whether a news article will trend on **Sina Weibo**. We analyze textual features, metadata, and engagement metrics to develop predictive models, optimizing their performance through various techniques, including **Bayesian optimization**.

## Data
- **Source**: Sina Weibo trending news (2021–2023)
- **Size**: 31,713 observations, 59 features
- **Target Variable**: Binary indicator (1 = trending, 0 = not trending)

## Methodology
- **Logistic Regression**: Baseline model with weighted class adjustment
- **Decision Trees**: Interpretable model for capturing feature importance
- **Random Forest**: Optimized using **Bayesian hyperparameter tuning**
- **Random Forest with PCA**: Dimensionality reduction for performance enhancement

## Model Evaluation
- Metrics: **Accuracy, Precision, Recall, F1-score, AUC (ROC Curve)**
- Best-performing model: **Random Forest** (AUC = 0.78, Accuracy = 66%)
- **Visualization**: Seaborn & Matplotlib used for class balance, ROC curves, and correlation heatmaps

## Results & Findings
- **Gini Importance Analysis**: Key features include word uniqueness ratio, sentiment polarity, and engagement metrics.
- **Random Forest outperformed other models**, achieving the best balance between recall and precision.
- **PCA reduced feature dimensionality** but slightly impacted classification performance.
