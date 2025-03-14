
# Alzheimer's Disease Classification

This repository contains machine learning models for classifying Alzheimer's disease using various supervised learning techniques. The dataset is preprocessed, analyzed, and evaluated using multiple classifiers.

## 📌 Features
- **Exploratory Data Analysis (EDA)**: Visualizing categorical and numerical feature distributions.
- **Feature Engineering**: Removing unnecessary columns, handling imbalanced data.
- **Data Preprocessing**: Normalization & Standardization using `MinMaxScaler` and `StandardScaler`.
- **Model Selection & Training**: Training multiple models and optimizing hyperparameters using `GridSearchCV`.
- **Performance Evaluation**: Confusion matrix, classification report, and accuracy comparison of models.

---

## 🛠️ Installation & Requirements

Ensure you have **Python 3.8+** installed. Install dependencies using:

```bash
pip install -r requirements.txt
```
Alternatively, manually install the required libraries:

```bash
 pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost
```
## 📂 File Structure
```bash
📂 Alzheimers-Disease-Classification
│-- 📄 Alzheimer's_Disease_Classification_.ipynb     # Main script for training models with output
│-- 📄 Dataset.py                                    # download dataset
│-- 📄 README.md                                     # Project documentation
│-- 📄 alzheimer's_disease_classification.py         # Main script
│--   requirements.txt                               # Dependencies
```


## Usage
Run the script:

```bash
python alzheimers_classification.py
```

## Models Used

* Decision Tree
* Random Forest
* Logistic Regression
* K-Nearest Neighbors
* Support Vector Machine
* XGBoost
* CatBoost

## Results
The best-performing model is displayed along with accuracy comparisons.



