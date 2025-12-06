# Appraisal Agent – AAI-590 Capstone Project - Group 12

### Housing Price Prediction Using Machine Learning and Deep Learning

This notebook contains the implementation of the **Appraisal Agent**, a core component of a larger multi-agent AI framework designed to automate and modernize stages of the U.S. mortgage process. This work was completed as part of the **AAI-590 Capstone** in the M.S. in Applied Artificial Intelligence program at the University of San Diego.

The Appraisal Agent predicts residential property values using machine learning and deep learning models. This includes data cleaning, exploratory analysis, model development, hyperparameter tuning, evaluation, and explainability using SHAP.

---

## Repository Contents

* **main-3.ipynb** – End-to-end workflow including preprocessing, EDA, model training, tuning, and SHAP analysis
* **README.md** – Project documentation

---

## Project Overview

The mortgage appraisal process is traditionally slow and highly manual. This project develops a data-driven appraisal engine capable of:

* Ingesting structured property data
* Predicting the market value of a home
* Providing interpretable model explanations
* Integrating into a multi-agent mortgage automation workflow

The Appraisal Agent corresponds to **Agent 3** within the broader system.

---

## Objectives

1. Load and preprocess Zillow housing data
2. Conduct exploratory data analysis
3. Train a baseline LightGBM model
4. Train a neural network model using TensorFlow/Keras
5. Tune LightGBM using RandomizedSearchCV
6. Evaluate all models using RMSE, MAE, and R²
7. Generate SHAP explainability visualizations
8. Select the best model for the Appraisal Agent

---

## Methods Used

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Gradient boosting using LightGBM
* Neural network modeling with dropout and early stopping
* Hyperparameter tuning
* Prevention of data leakage through train-only encoding and scaling
* SHAP model interpretability

---

## Technologies

Python
pandas
numpy
scikit-learn
LightGBM
TensorFlow / Keras
SHAP
matplotlib, seaborn
kagglehub
openpyxl

---

## Dataset

**Source:** Zillow.com House Price Prediction dataset (Kaggle)

The dataset contains approximately 7,000 properties with:

* Structural features (bedrooms, bathrooms, square footage, year built)
* Location attributes (city, county, latitude, longitude)
* Financial attributes
* Zillow-estimated features

Data preprocessing includes removal of high-missingness columns, elimination of leakage-prone fields, and preservation of essential predictors.

---

## Installation and Setup

1. Clone or download the repository
2. (Optional) Create and activate a virtual environment
3. Install required libraries
4. Launch Jupyter Notebook
5. Open and run **main-3.ipynb**

---

## Model Summary

### Baseline LightGBM Model

* Uses a training/validation split
* Employs early stopping
* Provides strong generalization performance

### Neural Network Model

* Feed-forward network
* ReLU activations
* Dropout regularization
* Early stopping based on validation loss

### Tuned LightGBM Model

* Hyperparameter optimization via RandomizedSearchCV
* Parameter search includes tree depth, leaf size, learning rate, sampling ratios, and regularization
* Improves training metrics but generalizes slightly worse than the baseline

---

## Evaluation

Models are evaluated using:

* RMSE
* MAE
* R²

A model comparison table is generated in the notebook showing performance across:

* Baseline LightGBM
* Tuned LightGBM
* Neural Network

The **baseline LightGBM** model performs best on unseen data.

---

## Explainability

The tuned LightGBM model is interpreted using **SHAP**.

Key SHAP insights include:

* Zillow-estimated features strongly influence predictions
* Square footage, bedrooms, and bathrooms are major structural predictors
* Location variables contribute meaningfully

---

## Final Model Selection

The **baseline LightGBM** model is selected as the final Appraisal Agent because:

* It generalizes best
* It avoids overfitting
* It trains efficiently
* It integrates smoothly with SHAP explainability
* It performs well on structured tabular data

---

## Future Enhancements

* Additional feature engineering (geospatial and temporal trends)
* Integration of external datasets
* Deployment as an API within a multi-agent workflow
* Evaluation of CatBoost and XGBoost
* End-to-end implementation of a production-ready appraisal pipeline

---

## Contributors

**AAI-590 Capstone Team**
* Zachary Artman
* Samantha Colbert
* Nabeel Khan
* Olga Pospelova

University of San Diego
Master of Science in Applied Artificial Intelligence
