import os
import time
from typing import Any, Dict, Optional

import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "rsi", "macd_hist", "ema50", "ema200",
    "bb_width", "atr_pct", "conf",
]


class MLGating:
    def __init__(self, csv_path: str = "logs/transactions.csv"):
        self.csv_path = csv_path
        self.model: Optional[Pipeline] = None
        self.last_train_ts = 0.0
        self.last_train_rows = 0
        self.last_metrics: Dict[str, Any] = {}

        self.min_rows = int(os.getenv("ML_MIN_ROWS", "40"))
        self.min_train_interval = int(os.getenv("ML_MIN_TRAIN_INTERVAL_SEC", "3600"))
        self.gate_threshold = float(os.getenv("ML_GATE_THRESHOLD", "0.25"))
        self.min_test_rows = int(os.getenv("ML_MIN_TEST_ROWS", "10"))

    def _safe_read_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.csv_path)
        except EmptyDataError:
            return pd.DataFrame()
        except Exception as e:
            print(f"[ML] read_csv error: {e}", flush=True)
            return pd.DataFrame()

        return df if not df.empty else pd.DataFrame()

    def _load_closed_trades(self) -> pd.DataFrame:
        df = self._safe_read_csv()
        if df.empty:
            return pd.DataFrame()

        needed = set(["channel", "event", "pnl_pct"] + FEATURES)
        missing = needed - set(df.columns)
        if missing:
            print(f"[ML] missing columns: {sorted(missing)}", flush=True)
            return pd.DataFrame()

        df = df.copy()
        df["channel"] = df["channel"].astype(str).str.strip().str.upper()
        df["event"] = df["event"].astype(str).str.strip().str.upper()

        df = df[
            (df["channel"] == "PROD") &
            (df["event"] == "TRADE_EXIT")
        ].copy()

        if df.empty:
            return pd.DataFrame()

        for col in FEATURES + ["pnl_pct"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # zachowujemy kolejność czasową, żeby test był na nowszych rekordach
        sort_col = None
        for candidate in ("ts_epoch", "ts_local", "ts_utc"):
            if candidate in df.columns:
                sort_col = candidate
                break
        if sort_col is not None:
            df = df.sort_values(sort_col).reset_index(drop=True)

        df = df.dropna(subset=FEATURES + ["pnl_pct"])
        return df

    def can_train(self, df: pd.DataFrame) -> bool:
        return len(df) >= self.min_rows

    def should_train_now(self, df: pd.DataFrame) -> bool:
        now = time.time()
        if not self.can_train(df):
            return False
        if (now - self.last_train_ts) < self.min_train_interval:
            return False
        if len(df) <= self.last_train_rows:
            return False
        return True

    def _build_xy(self, df: pd.DataFrame):
        X = df[FEATURES].astype(float)
        y = (df["pnl_pct"].astype(float) > 0.0).astype(int)
        return X, y

    def train_if_needed(self) -> Dict[str, Any]:
        df = self._load_closed_trades()

        if not self.can_train(df):
            return {
                "trained": False,
                "reason": f"too_few_rows ({len(df)}/{self.min_rows})",
            }

        if not self.should_train_now(df):
            return {
                "trained": False,
                "reason": f"cooldown_or_no_new_rows (rows={len(df)})",
            }

        X, y = self._build_xy(df)

        if y.nunique() < 2:
            return {
                "trained": False,
                "reason": "only_one_class_in_target",
            }

        split_idx = int(len(df) * 0.8)
        min_allowed_split = max(10, len(df) - self.min_test_rows)
        split_idx = min(split_idx, min_allowed_split)
        split_idx = max(split_idx, 1)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if len(X_test) == 0:
            return {
                "trained": False,
                "reason": "empty_test_split",
            }

        if y_train.nunique() < 2:
            return {
                "trained": False,
                "reason": "only_one_class_in_train",
            }

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        self.model = pipe
        self.last_train_ts = time.time()
        self.last_train_rows = len(df)
        self.last_metrics = {
            "rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "acc": float(acc),
            "winrate_train": float(y_train.mean()),
            "winrate_test": float(y_test.mean()),
        }

        return {
            "trained": True,
            **self.last_metrics,
        }

    def build_feature_row(self, features: Dict[str, Any]) -> Dict[str, float]:
        row: Dict[str, float] = {}
        for key in FEATURES:
            try:
                row[key] = float(features.get(key, 0.0))
            except (TypeError, ValueError):
                row[key] = 0.0
        return row

    def prob_win(self, features: Dict[str, Any]) -> Optional[float]:
        if self.model is None:
            return None

        row = self.build_feature_row(features)
        X = pd.DataFrame([row])
        return float(self.model.predict_proba(X)[0][1])

    def allow_trade(self, features: Dict[str, Any]) -> Dict[str, Any]:
        p = self.prob_win(features)
        if p is None:
            return {
                "model_ready": False,
                "prob_win": None,
                "threshold": self.gate_threshold,
                "allow": True,
                "reason": "model_not_trained",
            }

        allow = p >= self.gate_threshold
        return {
            "model_ready": True,
            "prob_win": float(p),
            "threshold": self.gate_threshold,
            "allow": allow,
            "reason": "passed" if allow else "below_threshold",
        }
