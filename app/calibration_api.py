import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# ---------- math utils ----------
EPS = 1e-6
def _clip(p): return np.clip(p, EPS, 1 - EPS)
def logit(p): p=_clip(p); return np.log(p) - np.log(1 - p)
def sigmoid(z): return 1 / (1 + np.exp(-z))

# ---------- single calibrator ----------
@dataclass
class CalibratorParams:
    method: str                   # "temperature" or "platt"
    T: float | None = None        # for temperature
    a: float | None = None        # for platt (slope)
    b: float | None = None        # for platt (bias)

class ConfidenceCalibrator:
    """
    One calibrator (global OR per-domain).
    method="temperature": single parameter T
    method="platt": logistic regression on logit(p)
    """
    def __init__(self, method: str = "temperature"):
        assert method in ("temperature", "platt")
        self.method = method
        self.params = CalibratorParams(method=method)
        self._lr = None  # sklearn model, only for platt

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        p_raw = _clip(np.asarray(p_raw, dtype=float))
        y = np.asarray(y, dtype=int)

        if self.method == "temperature":
            self.params.T = self._fit_temperature(p_raw, y)
        else:
            self._lr = self._fit_platt(p_raw, y)
            self.params.a = float(self._lr.coef_.ravel()[0])
            self.params.b = float(self._lr.intercept_.ravel()[0])
        return self

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = _clip(np.asarray(p_raw, dtype=float))
        if self.method == "temperature":
            if self.params.T is None:
                raise RuntimeError("Temperature scaler not fitted.")
            return sigmoid(logit(p_raw) / self.params.T)
        else:
            if self.params.a is None or self.params.b is None:
                raise RuntimeError("Platt scaler not fitted.")
            z = self.params.a * logit(p_raw) + self.params.b
            return sigmoid(z)

    # --- internals ---
    def _fit_temperature(self, p_raw, y, grid=(0.2, 5.0, 300)) -> float:
        z = logit(p_raw)
        lo, hi, steps = grid
        Ts = np.exp(np.linspace(np.log(lo), np.log(hi), steps))
        best_T, best_nll = 1.0, 1e99
        for T in Ts:
            p = sigmoid(z / T)
            nll = log_loss(y, _clip(p))
            if nll < best_nll:
                best_nll, best_T = nll, T
        return float(best_T)

    def _fit_platt(self, p_raw, y):
        X = logit(p_raw).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(X, y)
        return lr

    def to_dict(self) -> dict:
        return asdict(self.params)

    @classmethod
    def from_dict(cls, d: dict):
        cal = cls(method=d["method"])
        cal.params = CalibratorParams(**d)
        return cal


# ---------- manager for domains + IO ----------
class CalibrationManager:
    """
    Handles: loading CSVs, building P(y=1), fitting per-domain and global calibrators,
    saving/loading params, and serving calibrated confidences via `calibrate`.
    """
    def __init__(self, method: str = "temperature"):
        assert method in ("temperature", "platt")
        self.method = method
        self.global_cal = ConfidenceCalibrator(method=method)
        self.domain_cals: dict[str, ConfidenceCalibrator] = {}

    # ---- data loading helpers ----
    @staticmethod
    def _build_probs(
        df: pd.DataFrame,
        prob_col: str,
        label_col: str,
        pred_label_col: str | None = None,
        prob_is_pos_class: bool = True,
    ):
        """
        Returns p_raw (probabilities for y=1) and y (0/1).
        - If pred_label_col is provided and prob_is_pos_class=False,
          we assume 'prob_col' is confidence of the *predicted class* and we convert to P(y=1)
          by p = conf if pred_label==1 else (1-conf).
        - If prob_is_pos_class=True, we assume 'prob_col' already is P(y=1).
        """
        y = df[label_col].astype(int).to_numpy()
        conf = df[prob_col].astype(float).to_numpy()

        if pred_label_col and not prob_is_pos_class:
            pred = df[pred_label_col].astype(int).to_numpy()
            p_raw = np.where(pred == 1, conf, 1.0 - conf)
        else:
            p_raw = conf  # already P(y=1)

        return _clip(p_raw), y

    # ---- fitting ----
    def fit_from_csvs(
        self,
        domain_to_csv: dict[str, str],
        prob_col: str = "confidence",     # your file spelling
        label_col: str = "search_needed", # your gold label
        pred_label_col: str | None = None,
        prob_is_pos_class: bool = True,
    ):
        """
        Fit per-domain calibrators and a global calibrator from CSVs.
        Assumes each CSV has at least [prob_col, label_col].
        If you also have predicted labels and 'prob_col' is "confidence for predicted class",
        set pred_label_col="pred_label" (or whatever your column is) and prob_is_pos_class=False.
        """
        # Load per-domain
        domain_data = {}
        for dom, path in domain_to_csv.items():
            df = pd.read_csv(path)
            if prob_col not in df.columns or label_col not in df.columns:
                raise ValueError(f"{path} must have columns '{prob_col}' and '{label_col}'")
            p_raw, y = self._build_probs(df, prob_col, label_col, pred_label_col, prob_is_pos_class)
            domain_data[dom] = (p_raw, y)

        # Fit per-domain
        self.domain_cals = {}
        for dom, (p_raw, y) in domain_data.items():
            cal = ConfidenceCalibrator(self.method).fit(p_raw, y)
            self.domain_cals[dom] = cal

        # Fit global on union
        all_p = np.concatenate([v[0] for v in domain_data.values()])
        all_y = np.concatenate([v[1] for v in domain_data.values()])
        self.global_cal.fit(all_p, all_y)
        return self

    # ---- inference ----
    def calibrate(self, domain: str | None, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate a vector of probabilities (assumed to be P(y=1)).
        If domain calibrator is missing, falls back to global.
        """
        probs = _clip(np.asarray(probs, dtype=float))
        cal = self.domain_cals.get(domain, self.global_cal)
        return cal.calibrate(probs)

    def calibrate_confidence(self, domain: str, confidence: float) -> float:
        return float(np.asarray(self.calibrate(domain, confidence)).ravel()[0])

    # ---- persistence ----
    def save(self, path: str):
        obj = {
            "method": self.method,
            "global": self.global_cal.to_dict(),
            "domains": {k: v.to_dict() for k, v in self.domain_cals.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        mgr = cls(method=obj["method"])
        mgr.global_cal = ConfidenceCalibrator.from_dict(obj["global"])
        mgr.domain_cals = {k: ConfidenceCalibrator.from_dict(v) for k, v in obj["domains"].items()}
        return mgr
    
# ---- add this below the classes in calibration_api.py ----
import numpy as np
from sklearn.model_selection import train_test_split

def _ece(y_true, p_pred, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    p_pred = _clip(np.asarray(p_pred))
    bins = np.linspace(0,1,n_bins+1)
    e = 0.0; n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (p_pred >= lo) & (p_pred < hi) if i < n_bins-1 else (p_pred >= lo) & (p_pred <= hi)
        if not np.any(idx): 
            continue
        acc = y_true[idx].mean()
        conf = p_pred[idx].mean()
        e += (idx.sum()/n) * abs(acc - conf)
    return e

def _fit_select_one(p, y):
    """Return ('temperature' or 'platt', fitted_calibrator, metrics_dict)."""
    # split once (stratified)
    y = np.asarray(y).astype(int)
    p = _clip(np.asarray(p))
    Xtr, Xte, ytr, yte = train_test_split(p, y, test_size=0.2, stratify=y, random_state=42)

    # Temperature
    cal_T = ConfidenceCalibrator("temperature").fit(Xtr, ytr)
    pT = cal_T.calibrate(Xte)
    llT = log_loss(yte, _clip(pT))
    eT  = _ece(yte, pT)

    # Platt
    cal_P = ConfidenceCalibrator("platt").fit(Xtr, ytr)
    pP = cal_P.calibrate(Xte)
    llP = log_loss(yte, _clip(pP))
    eP  = _ece(yte, pP)

    # pick by log loss, tie-break ECE
    if (llT < llP) or (abs(llT-llP) < 1e-6 and eT <= eP):
        return "temperature", ConfidenceCalibrator("temperature").fit(p, y), {"logloss": llT, "ece": eT}
    else:
        return "platt", ConfidenceCalibrator("platt").fit(p, y), {"logloss": llP, "ece": eP}

def auto_fit_and_save(
    domain_to_csv: dict[str, str],
    out_path: str = "calibrators.json",
    prob_col: str = "confidence",
    label_col: str = "search_needed",
    pred_label_col: str | None = None,
    prob_is_pos_class: bool = True,
):
    """
    1) Loads each domain CSV
    2) Builds p_raw (as P(y=1)) and y
    3) Auto-selects method per domain (temperature vs platt) via 80/20 split
    4) Also selects a global method on pooled data
    5) Saves everything to out_path
    """
    # collect per-domain data
    per_dom = {}
    for dom, path in domain_to_csv.items():
        df = pd.read_csv(path)
        if prob_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"{path} must have '{prob_col}' and '{label_col}' columns")
        p_raw, y = CalibrationManager._build_probs(
            df, prob_col, label_col, pred_label_col, prob_is_pos_class
        )
        per_dom[dom] = (p_raw, y)

    # per-domain selection
    chosen = {}
    for dom, (p, y) in per_dom.items():
        m, cal, metrics = _fit_select_one(p, y)
        chosen[dom] = (m, cal, metrics)

    # global selection
    all_p = np.concatenate([v[0] for v in per_dom.values()])
    all_y = np.concatenate([v[1] for v in per_dom.values()])
    m_glob, cal_glob, glob_metrics = _fit_select_one(all_p, all_y)

    # build manager with chosen methods
    mgr = CalibrationManager(method=m_glob)
    mgr.global_cal = cal_glob
    mgr.domain_cals = {dom: cal for dom, (_, cal, _) in chosen.items()}

    # save
    mgr.save(out_path)

    # compact report
    report = {
        "global": {"method": m_glob, **glob_metrics},
        "domains": {dom: {"method": m, **metrics} for dom, (m, _, metrics) in chosen.items()}
    }
    return mgr, report

def load_manager(path: str) -> CalibrationManager:
    return CalibrationManager.load(path)

