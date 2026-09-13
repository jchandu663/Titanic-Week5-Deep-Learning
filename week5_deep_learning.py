"""
Week 5 Task: Deep Learning Application in Data Science
Dataset: Titanic
Framework: PyTorch
Task: Binary classification of passenger survival
Random seed: 42
"""

import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "data/titanic_data.csv"

def feature_engineer(df):
    d = df.copy()
    d["Title"] = d["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    d["Title"] = d["Title"].replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"})
    d.loc[~d["Title"].isin(["Mr","Miss","Mrs","Master"]), "Title"] = "Rare"
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["FarePerPerson"] = d["Fare"] / d["FamilySize"]
    d["Deck"] = d["Cabin"].fillna("U").str[0]

    features = ["Pclass","Sex","Age","SibSp","Parch","Fare",
                "Embarked","Title","FamilySize","IsAlone",
                "FarePerPerson","Deck"]
    return d[features], df["Survived"].astype(int)

df = pd.read_csv(DATA_PATH)
X, y = feature_engineer(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

numeric = ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","FarePerPerson"]
categorical = ["Sex","Embarked","Title","Deck"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_np = y_test.values

# Validation split from training data for early stopping
idx = np.arange(len(X_train))
train_idx, val_idx = train_test_split(
    idx, test_size=0.15, stratify=y_train.numpy().ravel(), random_state=SEED
)

class TitanicMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = TitanicMLP(X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

loader = DataLoader(
    TensorDataset(X_train[train_idx], y_train[train_idx]),
    batch_size=32, shuffle=True
)

best_state = None
best_val_loss = float("inf")
patience = 12
wait = 0

for epoch in range(1, 121):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(
            model(X_train[val_idx]), y_train[val_idx]
        ).item()

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    probability = torch.sigmoid(model(X_test)).numpy().ravel()

prediction = (probability >= 0.5).astype(int)

print("Week 5 Titanic Deep Learning Results")
print("------------------------------------")
print(f"Accuracy : {accuracy_score(y_test_np, prediction):.4f}")
print(f"Precision: {precision_score(y_test_np, prediction):.4f}")
print(f"Recall   : {recall_score(y_test_np, prediction):.4f}")
print(f"F1 Score : {f1_score(y_test_np, prediction):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test_np, probability):.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test_np, prediction, target_names=["Did Not Survive", "Survived"]
))

torch.save(model.state_dict(), "outputs/titanic_mlp_weights.pth")
