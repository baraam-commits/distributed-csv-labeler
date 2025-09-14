import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# ---------- math utils ----------
EPS = 1e-3  # slightly larger to tame extreme logits near 0/1
def _clip(p): 
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)

def logit(p):
    p = _clip(p)
    return np.log(p) - np.log(1 - p)

def sigmoid(z): 
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

# ---------- single calibrator ----------
@dataclass
class CalibratorParams:
    method: str                   # "temperature", "temperature_down", or "platt"
    T: float | None = None        # for temperature
    a: float | None = None        # for platt (slope)
    b: float | None = None        # for platt (bias)

ALLOW_PLATT = True #Has a tendency to bump up confidence far too much
TEMP_DOWN_PARAM = (1.0, 50.0, 500)
TEMP_PARAM = (0.2, 5.0, 300)
class ConfidenceCalibrator:
    """
    method="temperature_down"  -> temperature with T >= 1 (only pulls probs toward 0.5)
    method="temperature"       -> unconstrained temperature (T in [0.2,5])
    method="platt"             -> logistic regression on logit(p)
    """

    def __init__(self, method: str = "temperature_down"):
        assert method in ("temperature", "temperature_down", "platt")
        self.method = method
        self.params = CalibratorParams(method=method)
        self._lr: LogisticRegression | None = None  # only used for platt

    def fit(self, p_raw, y):
        p_raw = _clip(p_raw)
        y = np.asarray(y, dtype=int)

        if self.method == "temperature_down":
            self.params.T = self._fit_temperature(p_raw, y, grid=TEMP_DOWN_PARAM)  # T>=1 only
        elif self.method == "temperature":
            self.params.T = self._fit_temperature(p_raw, y, grid=TEMP_PARAM)
        else:  # platt
            self._lr = self._fit_platt(p_raw, y)
            self.params.a = float(self._lr.coef_.ravel()[0])
            self.params.b = float(self._lr.intercept_.ravel()[0])
        return self

    def calibrate(self, p_raw):
        p_raw = _clip(p_raw)
        if self.method in ("temperature", "temperature_down"):
            if self.params.T is None:
                raise RuntimeError("Temperature scaler not fitted.")
            return sigmoid(logit(p_raw) / self.params.T)
        else:  # platt
            if (self.params.a is None) or (self.params.b is None):
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
        lr = LogisticRegression(
            C=0.1, class_weight="balanced", solver="lbfgs", max_iter=1000
        )
        lr.fit(X, y)
        return lr

    def to_dict(self) -> dict:
        return asdict(self.params)

    @classmethod
    def from_dict(cls, d: dict):
        cal = cls(method=d["method"])
        cal.params = CalibratorParams(**d)
        # platt params are stored in params; we don't reconstruct sklearn model for inference
        return cal

# ---------- manager for domains + IO ----------
class CalibrationManager:
    def __init__(self, method: str = "temperature_down"):
        self.method = method
        self.global_cal = ConfidenceCalibrator(method=method)
        self.domain_cals: dict[str, ConfidenceCalibrator] = {}
        self.domain_shrink: dict[str | None, float] = {}  # domain -> lambda

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
          'prob_col' is confidence of the *predicted class*; convert to P(y=1)
          by p = conf if pred_label==1 else (1-conf).
        - If prob_is_pos_class=True, 'prob_col' already is P(y=1).
        """
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        if prob_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Missing required columns '{prob_col}' or '{label_col}'. Got: {df.columns.tolist()}")

        # Label -> 0/1 robustly
        lab = df[label_col].astype(str).str.strip().str.lower()
        mapping = {"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,"y":1,"n":0}
        y = pd.to_numeric(lab.map(mapping), errors="coerce")

        # Probabilities -> numeric in [0,1]
        conf = pd.to_numeric(df[prob_col], errors="coerce")

        mask = y.isin([0,1]) & conf.notna() & conf.between(0.0, 1.0, inclusive="both")
        kept = int(mask.sum()); dropped = int((~mask).sum())
        if dropped:
            print(f"[CAL] Dropping {dropped} invalid rows in '{label_col}'/'{prob_col}'")
        if kept == 0:
            raise ValueError("No valid rows after cleaning.")

        y = y[mask].astype(int).to_numpy()
        conf = conf[mask].to_numpy(dtype=float)

        if pred_label_col and not prob_is_pos_class:
            pred = pd.to_numeric(df.loc[mask, pred_label_col], errors="coerce")
            pred = pred.where(pred.isin([0, 1]))
            m2 = pred.notna()
            y = y[m2.to_numpy()]
            conf = conf[m2.to_numpy()]
            pred = pred[m2]
            p_raw = np.where(pred.to_numpy(dtype=int) == 1, conf, 1.0 - conf)
        else:
            p_raw = conf  # already P(y=1)

        return _clip(p_raw), y

    # ---- fitting from CSVs (optional path) ----
    def fit_from_csvs(
        self,
        domain_to_csv: dict[str, str],
        prob_col: str = "confidence",
        label_col: str = "search_needed",
        pred_label_col: str | None = None,
        prob_is_pos_class: bool = True,
    ):
        domain_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for dom, path in domain_to_csv.items():
            df = pd.read_csv(path)
            p_raw, y = self._build_probs(df, prob_col, label_col, pred_label_col, prob_is_pos_class)
            domain_data[dom] = (p_raw, y)

        # per-domain
        self.domain_cals = {}
        for dom, (p_raw, y) in domain_data.items():
            self.domain_cals[dom] = ConfidenceCalibrator(self.method).fit(p_raw, y)

        # global
        all_p = np.concatenate([v[0] for v in domain_data.values()])
        all_y = np.concatenate([v[1] for v in domain_data.values()])
        self.global_cal.fit(all_p, all_y)
        return self

    # ---- inference ----
    def _apply_shrink(self, p: np.ndarray, lam: float) -> np.ndarray:
        lam = float(np.clip(lam, 0.0, 1.0))
        if lam <= 1e-8:
            return p
        p = np.asarray(p, dtype=float)
        mask = (p < 1.0)  # keep exact 1.0 untouched
        p2 = p.copy()
        p2[mask] = (1.0 - lam) * p[mask] + lam * 0.5
        return p2

    def calibrate(self, domain: str | None, probs) -> np.ndarray:
        probs = _clip(probs)
        cal = self.domain_cals.get(domain, self.global_cal)
        out = cal.calibrate(probs)

        # --- NEW: never-up guard (except exact 1.0 preserved) ---
        out = np.asarray(out, dtype=float)
        probs_arr = np.asarray(probs, dtype=float)

        # keep exact 1.0 as exactly 1.0
        at_one = (probs_arr >= 1.0)
        out[at_one] = 1.0

        # for everything else, do not allow increases
        not_one = ~at_one
        out[not_one] = np.minimum(out[not_one], probs_arr[not_one])

        # domain shrink toward 0.5
        lam = self.domain_shrink.get(domain, 0.0)
        out = self._apply_shrink(out, lam)
        return out


    def calibrate_confidence(self, domain: str, confidence: float) -> float:
        if confidence == 1.0:
            return 1.0
        return float(np.asarray(self.calibrate(domain, confidence)).ravel()[0])

    # ---- persistence ----
    def save(self, path: str):
        obj = {
            "method": self.method,
            "global": self.global_cal.to_dict(),
            "domains": {k: v.to_dict() for k, v in self.domain_cals.items()},
            "domain_shrink": self.domain_shrink,
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
        mgr.domain_shrink = obj.get("domain_shrink", {})
        return mgr

# ---------- metrics & model selection ----------
def _ece(y_true, p_pred, n_bins: int = 10):
    y_true = np.asarray(y_true, dtype=int)
    p_pred = _clip(p_pred)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0; n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (p_pred >= lo) & (p_pred < hi) if i < n_bins - 1 else (p_pred >= lo) & (p_pred <= hi)
        if not np.any(idx):
            continue
        acc = y_true[idx].mean()
        conf = p_pred[idx].mean()
        e += (idx.sum() / n) * abs(acc - conf)
    return e

def _fit_select_one(p, y):
    """Return (method_name, fitted_calibrator, metrics_dict)."""
    y = np.asarray(y, dtype=int)
    p = _clip(p)
    n = len(y)
    if not ALLOW_PLATT:
        calT = ConfidenceCalibrator("temperature_down").fit(p, y)
        pT = calT.calibrate(p)
        return "temperature_down", calT, {"logloss": log_loss(y, _clip(pT)), "ece": _ece(y, pT), "bias": abs(pT.mean() - y.mean()), "note": "forced_tempdown"}
    classes = np.unique(y)

    # fallback if split impossible
    def _compare_on(arr_p, arr_y):
        calT = ConfidenceCalibrator("temperature_down").fit(arr_p, arr_y)
        pT = calT.calibrate(arr_p); llT = log_loss(arr_y, _clip(pT)); eT = _ece(arr_y, pT)
        calP = ConfidenceCalibrator("platt").fit(arr_p, arr_y)
        pP = calP.calibrate(arr_p); llP = log_loss(arr_y, _clip(pP)); eP = _ece(arr_y, pP)
        prev = arr_y.mean()
        biasT = abs(pT.mean() - prev); biasP = abs(pP.mean() - prev)
        lam = 5.0
        scoreT = llT + lam * (biasT ** 2)
        scoreP = llP + lam * (biasP ** 2)
        if scoreT <= scoreP:
            return "temperature_down", calT, {"logloss": llT, "ece": eT, "bias": biasT, "note": "no_split"}
        else:
            return "platt", calP, {"logloss": llP, "ece": eP, "bias": biasP, "note": "no_split"}

    if (n < 10) or (len(classes) < 2):
        return _compare_on(p, y)

    try:
        Xtr, Xte, ytr, yte = train_test_split(p, y, test_size=0.2, stratify=y, random_state=42)
    except Exception:
        return _compare_on(p, y)

    # evaluate on held-out
    calT = ConfidenceCalibrator("temperature_down").fit(Xtr, ytr)
    pT = calT.calibrate(Xte); llT = log_loss(yte, _clip(pT)); eT = _ece(yte, pT); biasT = abs(pT.mean() - y.mean())
    calP = ConfidenceCalibrator("platt").fit(Xtr, ytr)
    pP = calP.calibrate(Xte); llP = log_loss(yte, _clip(pP)); eP = _ece(yte, pP); biasP = abs(pP.mean() - y.mean())

    lam = 5.0  # penalize deviation from prevalence (calibration-in-the-large)
    scoreT = llT + lam * (biasT ** 2)
    scoreP = llP + lam * (biasP ** 2)

    if scoreT <= scoreP:
        return "temperature_down", ConfidenceCalibrator("temperature_down").fit(p, y), {"logloss": llT, "ece": eT, "bias": biasT}
    else:
        return "platt", ConfidenceCalibrator("platt").fit(p, y), {"logloss": llP, "ece": eP, "bias": biasP}

# ---------- high-level API ----------
def auto_fit_and_save(
    domain_to_csv: dict[str, str],
    out_path: str = r"app/app_data/Calibration_data/calibrators.json",
    prob_col: str = "confidence",
    label_col: str = "search_needed",
    pred_label_col: str | None = None,
    prob_is_pos_class: bool = True,
):
    """
    1) Load each domain CSV and build (p_raw, y)
    2) Auto-select method per domain (temperature_down vs platt) via held-out split
    3) Do the same for global
    4) Compute per-domain shrink lambda to hit target means
    5) Save manager to out_path and return (mgr, report)
    """
    # collect per-domain data
    per_dom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dom, path in domain_to_csv.items():
        df = pd.read_csv(path)
        p_raw, y = CalibrationManager._build_probs(df, prob_col, label_col, pred_label_col, prob_is_pos_class)
        per_dom[dom] = (p_raw, y)

    # per-domain selection
    chosen: dict[str, tuple[str, ConfidenceCalibrator, dict]] = {}
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

    # target means (accept both spellings for programming)
    TARGET = {"programming": 0.64, "programing": 0.64, "general": 0.56}

    # compute per-domain shrink λ to hit targets (preserving 1.0)
    mgr.domain_shrink = {}
    for dom, (p, _y) in per_dom.items():
        cal = chosen[dom][1]
        p_cal = np.asarray(cal.calibrate(_clip(p)), dtype=float)
        mean_cal = float(np.mean(p_cal))
        target = TARGET.get(dom, 0.58)
        if mean_cal > 0.5:
            lam = np.clip((mean_cal - target) / max(mean_cal - 0.5, 1e-6), 0.0, 1.0)
        else:
            lam = 0.0
        mgr.domain_shrink[dom] = float(lam)

    # global fallback shrink (optional)
    p_all_cal = np.concatenate([mgr.domain_cals[dom].calibrate(_clip(per_dom[dom][0])) for dom in per_dom.keys()])
    mean_all = float(np.mean(p_all_cal))
    if mean_all > 0.5:
        lam_g = np.clip((mean_all - 0.58)/max(mean_all - 0.5, 1e-6), 0.0, 1.0)
    else:
        lam_g = 0.0
    mgr.domain_shrink[None] = float(lam_g)

    # save
    mgr.save(out_path)

    # compact report
    report = {
        "global": {"method": m_glob, **glob_metrics, "shrink_lambda": mgr.domain_shrink.get(None, 0.0)},
        "domains": {
            dom: {"method": m, **metrics, "shrink_lambda": mgr.domain_shrink.get(dom, 0.0)}
            for dom, (m, _, metrics) in chosen.items()
        }
    }
    return mgr, report

def load_manager(path: str) -> CalibrationManager:
    return CalibrationManager.load(path)
