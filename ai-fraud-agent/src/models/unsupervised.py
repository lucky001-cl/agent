# src/models/unsupervised.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class UnsupervisedModel:
    def __init__(self, model: IsolationForest = None, feature_cols: list = None):
        self.model = model
        self.feature_cols = feature_cols or []

    @staticmethod
    def train_from_csv(csv_path: str, save_path: str = "isoforest.pkl", feature_cols: list = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")
        df = pd.read_csv(csv_path)
        # If no feature cols specified, take numeric columns
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[feature_cols].fillna(0.0).values
        model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
        model.fit(X)
        inst = UnsupervisedModel(model=model, feature_cols=feature_cols)
        inst.save(save_path)
        return inst

    def score(self, X: np.ndarray):
        """
        IsolationForest.score_samples returns higher for more normal samples.
        We convert to anomaly score in 0..1 where higher is more anomalous.
        """
        raw = self.model.score_samples(X)  # higher = more normal
        # Normalize raw to 0..1 then invert
        minv = raw.min()
        maxv = raw.max()
        if maxv - minv == 0:
            norm = np.ones_like(raw) * 0.5
        else:
            norm = (raw - minv) / (maxv - minv)
        return 1.0 - norm

    def score_dict(self, feat_dict: dict) -> float:
        import numpy as np
        x = np.array([feat_dict.get(c, 0.0) for c in self.feature_cols], dtype=float).reshape(1, -1)
        s = float(self.score(x)[0])
        return s

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_cols": self.feature_cols}, f)

    @staticmethod
    def load(path: str):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = UnsupervisedModel(model=data["model"], feature_cols=data["feature_cols"])
        return inst
