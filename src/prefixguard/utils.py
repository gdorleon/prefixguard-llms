# src/prefixguard/utils.py
# A small bag of helpers for data loading, metrics, obfuscation, and calibration.
# The comments are written like you would explain things to a teammate.

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Every CSV must have these columns. Keep it simple and explicit.
REQUIRED_COLS = ["text", "label"]

def load_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV and make sure the required columns are present.
    We do not force label dtype here because mapping is handled in the train script.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {missing}")
    return df

def label_stats(df: pd.DataFrame) -> Dict:
    """
    Tiny summary for logging. Helpful sanity check before training.
    """
    counts = df["label"].value_counts().to_dict()
    return {"num_examples": len(df), "label_counts": counts}

# ----------------------------- Obfuscation -----------------------------------
# These transformations simulate noisy social media text.
# LEET_MAP replaces letters with lookalike digits.
# HOMOGLYPHS uses Cyrillic or Greek characters that look like Latin ones.

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "b": "8", "g": "9"}
HOMOGLYPHS = {
    "a": "а",  # Cyrillic a
    "e": "е",  # Cyrillic e
    "o": "ο",  # Greek omicron
    "p": "р",  # Cyrillic er
    "c": "с",  # Cyrillic es
    "x": "х",  # Cyrillic ha
    "y": "у",  # Cyrillic u
}

def obfuscate_text(s: str, mode: str = "mixed") -> str:
    """
    Apply a simple obfuscation to a string.
    - mode 'leet' picks from LEET_MAP only
    - mode 'homo' picks from HOMOGLYPHS only
    - mode 'mixed' randomly mixes both with some probability
    The goal is to stress test robustness, not to be perfect.
    """
    out = []
    for ch in str(s):
        lower = ch.lower()
        if mode == "leet" and lower in LEET_MAP and ch.isalpha():
            out.append(LEET_MAP[lower])
        elif mode == "homo" and lower in HOMOGLYPHS and ch.isalpha():
            repl = HOMOGLYPHS[lower]
            out.append(repl.upper() if ch.isupper() else repl)
        elif mode == "mixed":
            rnd = np.random.rand()
            if lower in LEET_MAP and ch.isalpha() and rnd < 0.4:
                out.append(LEET_MAP[lower])
            elif lower in HOMOGLYPHS and ch.isalpha() and 0.4 <= rnd < 0.7:
                repl = HOMOGLYPHS[lower]
                out.append(repl.upper() if ch.isupper() else repl)
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)

def make_obfuscated_df(df: pd.DataFrame, mode: str = "mixed") -> pd.DataFrame:
    """
    Copy the dataframe and obfuscate the text column. Labels remain untouched.
    """
    obf = df.copy()
    obf["text"] = obf["text"].astype(str).apply(lambda s: obfuscate_text(s, mode=mode))
    return obf

# ------------------------------- Fairness ------------------------------------
# We compute per group metrics when a column named 'group' is present.
# For F1 we support binary and multiclass. For FPR gap we only report a value
# when the task is binary with classes {0, 1}. Otherwise we return a note.

def _is_binary_labels(y: np.ndarray) -> bool:
    vals = np.unique(y)
    return len(vals) == 2 and set(vals.tolist()) == {0, 1}

def per_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: List[str]) -> Dict:
    """
    Compute per group F1 and gaps. If labels are binary 0 or 1
    we also compute FPR per group and its gap.

    Returns a dict with:
      - per_group_f1
      - worst_group_f1
      - f1_gap
      - per_group_fpr and fpr_gap if binary, else an explanatory note
    """
    res: Dict = {}
    unique_groups = sorted(set(groups))
    f1s: List[Tuple[str, float]] = []
    fprs: List[Tuple[str, float]] = []

    is_binary = _is_binary_labels(y_true)

    for g in unique_groups:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if not idx:
            continue
        y_t = y_true[idx]
        y_p = y_pred[idx]

        # F1 for this group
        # Use binary average when the task is binary, else macro within the group
        avg = "binary" if is_binary else "macro"
        try:
            f1_g = f1_score(y_t, y_p, average=avg, pos_label=1 if is_binary else None)
        except ValueError:
            # In rare cases a class may be missing in this subset. Fall back to accuracy.
            f1_g = accuracy_score(y_t, y_p)
        f1s.append((g, float(f1_g)))

        # FPR only makes sense in the classic binary setup
        if is_binary:
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            denom = (fp + tn) if (fp + tn) > 0 else 1
            fpr_g = fp / denom
            fprs.append((g, float(fpr_g)))

    if not f1s:
        return {"note": "no groups found"}

    # Aggregate F1 stats
    f1_values = [v for _, v in f1s]
    res["per_group_f1"] = {g: v for g, v in f1s}
    res["worst_group_f1"] = float(min(f1_values))
    res["f1_gap"] = float(max(f1_values) - min(f1_values))

    # Aggregate FPR stats when available
    if fprs:
        fpr_values = [v for _, v in fprs]
        res["per_group_fpr"] = {g: v for g, v in fprs}
        res["fpr_gap"] = float(max(fpr_values) - min(fpr_values))
    else:
        res["per_group_fpr"] = {"note": "FPR reported only for binary labels 0 and 1"}
        res["fpr_gap"] = None

    return res

# ----------------------------- Calibration -----------------------------------
# ECE is a scalar that measures how close predicted confidence is to actual accuracy.
# We support both binary and multiclass by accepting predicted confidences and
# optionally predicted class ids. If preds is given, we compute correctness based
# on match between preds and labels. If preds is None, we assume binary and use
# a 0.5 threshold on the confidence vector.

def expected_calibration_error(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    preds: Optional[np.ndarray] = None
) -> float:
    """
    Compute 10 bin ECE by default.

    Inputs
      confidences: shape (N,) with the model confidence for the chosen class
                   For multiclass use max softmax per example.
      labels: shape (N,) with integer gold labels
      n_bins: number of bins across [0, 1]
      preds: optional shape (N,) predicted labels. If provided we use correctness
             = (preds == labels). If not provided we assume binary and derive
             correctness from confidences >= 0.5.

    Returns
      scalar ECE in [0, 1]
    """
    confidences = np.asarray(confidences).astype(float)
    labels = np.asarray(labels)

    if preds is not None:
        preds = np.asarray(preds)
        correct = (preds == labels).astype(float)
    else:
        # Binary fallback. This keeps backward compatibility with older calls.
        correct = (labels == (confidences >= 0.5)).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidences)
    for i in range(n_bins):
        m = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if not np.any(m):
            continue
        conf_bin = float(np.mean(confidences[m]))
        acc_bin = float(np.mean(correct[m]))
        ece += (np.sum(m) / N) * abs(acc_bin - conf_bin)
    return float(ece)

def temperature_scale(logits: np.ndarray, labels: np.ndarray, init_T: float = 1.0) -> float:
    """
    Learn a single temperature parameter T by minimizing negative log likelihood
    on a development set. Works for binary or multiclass because it uses softmax.
    """
    from scipy.optimize import minimize

    def nll(T):
        T = max(1e-3, T)
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)   # prevent overflow
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        y = labels.astype(int)
        eps = 1e-12
        return -np.mean(np.log(p[np.arange(len(y)), y] + eps))

    res = minimize(lambda t: nll(float(t)), x0=[init_T], method="L-BFGS-B", bounds=[(1e-3, 100.0)])
    return float(res.x[0])

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Stable softmax for numpy arrays shaped (N, C).
    """
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Minimal report that matches what we log elsewhere.
    For multiclass we use macro average. For binary it is the same behavior.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return {"macro_f1": float(macro_f1), "accuracy": float(acc)}

