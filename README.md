# AI-Driven Credit Risk Analytics System

## 1. Problem Statement

Financial institutions must evaluate borrower creditworthiness accurately to reduce loan default risk while ensuring fair lending decisions. Traditional credit assessment processes are often manual, time-consuming, and subjective.

The objective of this project is to design and implement a machine learning–based credit risk scoring system that predicts whether a borrower is likely to default on a loan using historical borrower and loan data. The system provides clear and interpretable predictions through a simple web-based user interface.

---

## 2. Dataset Description

- **Dataset Name:** Credit Risk Benchmark Dataset  
- **Source:** Kaggle  
- **Format:** CSV  

The dataset contains historical information related to borrowers and loans, which is used to predict credit default risk.

### Example Features
- Borrower income  
- Employment status  
- Loan amount  
- Loan tenure  
- Credit history length  
- Past defaults  

### Target Variable
- **default**
  - `0` → No Default (Low Risk)
  - `1` → Default (High Risk)

Dataset location:
```
data/raw/Credit Risk Benchmark Dataset.csv
```

---

## 3. Machine Learning Pipeline

```
Raw Data
  ↓
Data Preprocessing
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Deployment (UI)
```

### Preprocessing Steps
- Missing value handling  
- Categorical encoding  
- Feature scaling  

### Model Used
- **Logistic Regression**

---

## 4. Model Evaluation

The model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- ROC-AUC Score  

These metrics help assess model reliability and risk prediction performance.

---

## 5. How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run preprocessing
```bash
python src/preprocessing.py
```

### Step 3: Train the model
```bash
python src/train_model.py
```

### Step 4: Run the UI
```bash
streamlit run app.py
```

---

## 6. Project Structure

```
credit-risk-ai/
│
├── data/
│   ├── raw/Credit Risk Benchmark Dataset.csv
│   └── processed/
├── notebooks/
    ├── data_understanding.ipynb
    ├── model_training.ipynb
    └── preprocessingn.ipynb
├── src/
├── models/
├── app.py
├── requirements.txt
└── README.md
```

---

## 7. Conclusion

This project automates credit risk prediction using machine learning and provides an end-to-end pipeline from data preprocessing to model deployment with a user interface.
