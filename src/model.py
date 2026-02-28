from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib


def _to_builtin(value: Any) -> Any:
    # Numpy / pandas scalars often have .item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _norm(s: str) -> str:
    return " ".join(s.strip().split()).lower()


CONTRACT_TYPE_CANONICAL = {
    "Month-to-Month": 0,
    "One Year": 1,
    "Two Year": 2,
}
CONTRACT_TYPE_MAP = {_norm(k): v for k, v in CONTRACT_TYPE_CANONICAL.items()}

INTERNET_TYPE_CANONICAL = {
    "Cable": 0,
    "DSL": 1,
    "Fiber Optic": 2,
    "None": 3,
}
INTERNET_TYPE_MAP = {_norm(k): v for k, v in INTERNET_TYPE_CANONICAL.items()}


class Model:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Use env in Docker so /flask-app/model_artifacts doesn't shadow this module
            model_dir = os.environ.get("CHURN_MODEL_DIR", "model")
            model_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    model_dir,
                    "final_model.pkl",
                )
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Expected a pickled sklearn/CatBoost pipeline saved to model/final_model.pkl."
            )

        self.model_path = model_path
        self.model = joblib.load(self.model_path)

    def predict(self, input_features):
        return self.model.predict(input_features)

    def _feature_names(self) -> Optional[List[str]]:
        # CatBoostClassifier usually supports one of these.
        names = None
        if hasattr(self.model, "feature_names_"):
            try:
                names = list(getattr(self.model, "feature_names_"))
            except Exception:
                names = None
        if not names and hasattr(self.model, "get_feature_names"):
            try:
                names = list(self.model.get_feature_names())
            except Exception:
                names = None
        return names or None

    def _preprocess_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a user-friendly payload (strings) into the numeric representation the
        trained model expects.
        """
        out = dict(payload)

        if "contract_type" in out and isinstance(out["contract_type"], str):
            key = _norm(out["contract_type"])
            if key not in CONTRACT_TYPE_MAP:
                raise ValueError(
                    "Unknown contract_type. Use one of: "
                    + ", ".join(CONTRACT_TYPE_CANONICAL.keys())
                )
            out["contract_type"] = CONTRACT_TYPE_MAP[key]

        if "internet_type" in out and isinstance(out["internet_type"], str):
            key = _norm(out["internet_type"])
            if key not in INTERNET_TYPE_MAP:
                raise ValueError(
                    "Unknown internet_type. Use one of: "
                    + ", ".join(INTERNET_TYPE_CANONICAL.keys())
                )
            out["internet_type"] = INTERNET_TYPE_MAP[key]

        # Best-effort numeric casting for common numeric fields
        for k in ("num_referrals", "num_dependents", "total_monthly_fee", "tenure_months", "age"):
            if k in out and isinstance(out[k], str):
                try:
                    out[k] = float(out[k]) if "." in out[k] else int(out[k])
                except Exception:
                    raise ValueError(f"Field '{k}' must be numeric.")

        return out

    def _row_from_payload(self, payload: Any) -> Tuple[Sequence[Any], Optional[List[str]]]:
        """
        Returns a single row (list/tuple) plus the feature order used (if any).
        """
        if isinstance(payload, (list, tuple)):
            return payload, None

        if not isinstance(payload, dict) or not payload:
            raise ValueError(
                "Payload must be either (a) a non-empty JSON object {feature: value} "
                "or (b) a JSON array of feature values [v1, v2, ...]."
            )

        payload = self._preprocess_payload(payload)
        feature_order = self._feature_names()
        if not feature_order:
            raise ValueError(
                "This saved model does not expose feature names. "
                "Send a JSON array of feature values in the exact training order, "
                "or update `src/model.py` to hardcode FEATURE_ORDER."
            )

        missing = [f for f in feature_order if f not in payload]
        if missing:
            raise KeyError(f"Missing required feature(s): {', '.join(missing)}")

        row = [payload[f] for f in feature_order]
        return row, feature_order

    def predict_one(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept a JSON object (feature_name -> value) and return a JSON-safe result.
        For CatBoost, this uses the model's stored feature names to build the row.
        """
        row, _feature_order = self._row_from_payload(payload)

        y_pred = self.model.predict([row])
        pred_value = _to_builtin(y_pred[0] if hasattr(y_pred, "__len__") else y_pred)

        result: Dict[str, Any] = {"prediction": pred_value}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([row])
            if proba is not None and hasattr(proba, "__len__") and len(proba) > 0:
                row = proba[0]
                if hasattr(row, "__len__") and len(row) >= 2:
                    result["churn_probability"] = float(_to_builtin(row[1]))
                elif hasattr(row, "__len__") and len(row) == 1:
                    result["probability"] = float(_to_builtin(row[0]))

        return result

    def predict_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict_one(p) for p in payloads]

    def schema(self) -> Dict[str, Any]:
        return {
            "features": self._feature_names(),
            "mappings": {
                "contract_type": CONTRACT_TYPE_CANONICAL,
                "internet_type": INTERNET_TYPE_CANONICAL,
            },
        }

