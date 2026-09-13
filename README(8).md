# Week 5 — Deep Learning Application in Data Science 🧠🚢

## Project
**Titanic Survival Prediction using a Neural Network (PyTorch)**

This Week 5 submission extends the Titanic work from the earlier EDA, clustering, and supervised-learning tasks into a practical deep-learning application. The problem is formulated as **binary classification**: predict whether a passenger survived.

### 🎯 Objectives
- Select a suitable public dataset.
- Define a clear classification problem.
- Engineer and preprocess features.
- Design a neural network architecture.
- Train the network using PyTorch.
- Evaluate it with accuracy, precision, recall, F1-score and ROC-AUC.
- Analyze overfitting, resource constraints and model limitations.
- Document the complete process with diagrams, code and visual results.

## 📊 Dataset
- Dataset: Titanic passenger dataset
- Rows: **891**
- Original columns: **12**
- Target: `Survived`
- Survival rate: **38.38%**
- Missing Age values: **177**
- Missing Cabin values: **687**
- Missing Embarked values: **2**

## 🧩 Features
Feature engineering adds:
- `Title`
- `FamilySize`
- `IsAlone`
- `FarePerPerson`
- `Deck`

Numerical features are imputed and standardized. Categorical features are imputed and one-hot encoded. The resulting neural-network input contains **27 encoded features**.

## 🏗️ Neural Network
```text
Input (27 features)
        ↓
Dense 64 + ReLU
        ↓
Dropout 0.25
        ↓
Dense 32 + ReLU
        ↓
Dropout 0.20
        ↓
Dense 16 + ReLU
        ↓
Dense 1 + Sigmoid probability
```

**Optimizer:** Adam  
**Learning rate:** 0.001  
**Batch size:** 32  
**Loss:** Binary Cross-Entropy with logits  
**Seed:** 42  
**Early stopping:** validation-loss based

## 📈 Test Results
| Metric | Score |
|---|---:|
| Accuracy | 0.8212 |
| Precision | 0.8033 |
| Recall | 0.7101 |
| F1 Score | 0.7538 |
| ROC-AUC | 0.8534 |
| Epochs trained | 26 |

## 📁 Repository Structure
```text
Week5_Titanic_Deep_Learning/
├── data/
│   └── titanic_data.csv
├── outputs/
│   ├── architecture_diagram.png
│   ├── workflow_diagram.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── performance_metrics.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── target_distribution.png
│   ├── dataset_summary.csv
│   ├── classification_report.txt
│   ├── metrics.json
│   ├── test_predictions.csv
│   └── titanic_mlp_model.pt
├── notebook/
│   └── Week5_Titanic_Deep_Learning.ipynb
├── docs/
│   ├── Week5_Titanic_Deep_Learning_Report.docx
│   └── submission_description_200_words.txt
├── week5_deep_learning.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run
```bash
python -m pip install -r requirements.txt
python week5_deep_learning.py
```

For the notebook:
```bash
jupyter notebook
```
Open `notebook/Week5_Titanic_Deep_Learning.ipynb` and run all cells.

## 🔬 Key Findings
The model reached **82.12% test accuracy** and **0.853 ROC-AUC**. The result shows that a compact neural network can learn useful nonlinear relationships from engineered Titanic passenger features. Dropout and early stopping were included to reduce overfitting.

## ⚠️ Limitations
- The dataset is relatively small for deep learning.
- Results depend on the train/test split and random seed.
- The Titanic dataset is historical and should not be treated as a modern operational prediction system.
- A larger dataset and systematic hyperparameter search would provide stronger evidence.

## 🚀 Future Scope
- Hyperparameter tuning with multiple seeds.
- Batch normalization and learning-rate scheduling.
- Class weighting or focal loss.
- Comparison with deeper/wider MLPs and tree-based models.
- Explainability using SHAP or permutation methods.
- Deployment as a small Streamlit/Flask demonstration.

## 📚 Technical Reference
The architecture follows the standard Keras/PyTorch-style neural-network workflow: preprocess data, stack dense layers, train with an optimizer, evaluate on held-out data, and monitor validation performance. Dropout is used as a regularization technique to reduce overfitting. TensorFlow documentation describes Sequential dense-layer stacking and dropout-based regularization. 
