A
b
# Iconnect-task: Iris Classification Pipeline

This project implements a **complete machine learning pipeline** for classifying the Iris dataset. It demonstrates how to structure a Python project with multiple modules, handle preprocessing, train different models, and evaluate them effectively.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Features](#features)  
- [Dependencies](#dependencies)  
- [Usage](#usage)  
- [Pipeline Overview](#pipeline-overview)  
- [Model Evaluation](#model-evaluation)  
- [Notes](#notes)  

---

## Project Structure

 ```

Iconnect-task/
│
├── main.py                     # Main script to run the pipeline
├── models/
│   ├── **init**.py
│   ├── pipeline_lr.py          # Logistic Regression pipeline
│   ├── pipeline_rf.py          # Random Forest pipeline with GridSearchCV
│   └── pipeline_svm.py         # SVM pipeline with GridSearchCV
├── dataOperation/
│   ├── **init**.py
│   └── data_loader.py          # Loads the Iris dataset
│   └── preprocessing.py        # Functions for missing values, outliers, normalization
├── README.md

```

- Each folder is a Python **package** thanks to `__init__.py`.  
- Modules can be imported across folders using proper package paths.

---

## Features

- Load Iris dataset using `scikit-learn`
- Handle missing values automatically
- Remove outliers using IQR
- Normalize features with `StandardScaler`
- Train models:
  - Logistic Regression
  - Random Forest with hyperparameter tuning
  - SVM with hyperparameter tuning
- Evaluate models with:
  - Accuracy
  - Classification report
  - Confusion matrix visualization

---

## Dependencies

- Python >= 3.8  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

Install dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
````

---

## Usage

1. Make sure you are in the project root directory (`Iconnect-task/`).
2. Run any pipeline module using Python **module mode**:

```bash
python -m models.pipeline_rf
```
or 
```bash
python -m models.pipeline_svm
```


* This ensures that Python correctly finds all packages.



---

## Pipeline Overview

1. **Data Loading**

```python
from dataOperation.data_loader import load_data
X, y = load_data()
```

2. **Preprocessing**

```python
from preprocessing.preprocessing import handle_missing_values, remove_outliers, normalize_data
X = handle_missing_values(X)
X, y = remove_outliers(X, y)
X_scaled = normalize_data(X)
```

3. **Train-Test Split**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

4. **Model Training & Hyperparameter Tuning**

* Random Forest and SVM pipelines use **GridSearchCV** to find the best parameters.
* Logistic Regression is trained as a baseline.

---

## Model Evaluation

* **Accuracy:** Overall model correctness
* **Classification Report:** Precision, recall, F1-score per class
* **Confusion Matrix:** Visual representation of predictions vs actual labels

Example:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

acc = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()
```

---

## Notes

* Always run scripts from the **project root** to avoid import errors.
* Make sure all packages have `__init__.py` to enable proper imports.

---

## Author

Yaseen Ashqar





هل تحب أسويها لك؟
```
