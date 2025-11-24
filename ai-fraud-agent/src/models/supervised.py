# src/models/supervised.py
import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class SupervisedModel:
    def __init__(self, model: XGBClassifier = None, feature_cols: list = None):
        self.model = model
        self.feature_cols = feature_cols or []

    @staticmethod
    def train_from_csv(csv_path: str, label_col: str = "is_fraud", save_path: str = "xgboost_model.pkl", feature_cols: list = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")
        df = pd.read_csv(csv_path)
        if label_col not in df.columns:
            raise ValueError(f"Label column {label_col} not in CSV")
        if feature_cols is None:
            # use numeric columns except label
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != label_col]
        X = df[feature_cols].fillna(0.0).values
        y = df[label_col].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)
        model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1]) if len(set(y_val))>1 else 0.0
        inst = SupervisedModel(model=model, feature_cols=feature_cols)
        inst.save(save_path)
        print(f"[TRAIN] Trained XGBoost model AUC={auc:.4f} saved to {save_path}")
        return inst

    def predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)[:,1]

    def predict_proba_dict(self, feat_dict: dict) -> float:
        import numpy as np
        X = np.array([feat_dict.get(c, 0.0) for c in self.feature_cols], dtype=float).reshape(1,-1)
        return float(self.predict_proba(X)[0])

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_cols": self.feature_cols}, f)

    @staticmethod
    def load(path: str):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = SupervisedModel(model=data["model"], feature_cols=data["feature_cols"])
        return inst
