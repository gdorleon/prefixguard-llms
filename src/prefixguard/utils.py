import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

REQUIRED_COLS = ["text", "label"]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {missing}")
    return df

def label_stats(df: pd.DataFrame) -> Dict:
    counts = df["label"].value_counts().to_dict()
    return {"num_examples": len(df), "label_counts": counts}

# ---- Obfuscation ----
LEET_MAP = {"a":"4","e":"3","i":"1","o":"0","s":"5","t":"7","b":"8","g":"9"}
HOMOGLYPHS = {"a":"а","e":"е","o":"ο","p":"р","c":"с","x":"х","y":"у"}  # Cyrillic/Greek lookalikes

def obfuscate_text(s: str, mode: str = "mixed") -> str:
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

def make_obfuscated_df(df: pd.DataFrame, mode: str="mixed") -> pd.DataFrame:
    obf = df.copy()
    obf["text"] = obf["text"].astype(str).apply(lambda s: obfuscate_text(s, mode=mode))
    return obf

# ---- Fairness ----
def per_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: List[str]) -> Dict:
    res = {}
    unique_groups = sorted(list(set(groups)))
    f1s, fprs = [], []
    for g in unique_groups:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if not idx:
            continue
        y_t = y_true[idx]
        y_p = y_pred[idx]
        f1_g = f1_score(y_t, y_p, average="binary", pos_label=1)

        cm = confusion_matrix(y_t, y_p, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        denom = (fp + tn) if (fp + tn) > 0 else 1
        fpr_g = fp / denom

        f1s.append((g, f1_g))
        fprs.append((g, fpr_g))

    if not f1s:
        return {"note": "no groups found"}

    f1_values = [v for _, v in f1s]
    f1_gap = float(max(f1_values) - min(f1_values))
    worst_group_f1 = float(min(f1_values))

    fpr_values = [v for _, v in fprs]
    fpr_gap = float(max(fpr_values) - min(fpr_values))

    res["per_group_f1"] = {g: float(v) for g, v in f1s}
    res["worst_group_f1"] = worst_group_f1
    res["f1_gap"] = f1_gap
    res["per_group_fpr"] = {g: float(v) for g, v in fprs}
    res["fpr_gap"] = fpr_gap
    return res

# ---- Calibration ----
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i+1])
        if not np.any(m):
            continue
        conf = np.mean(probs[m])
        acc  = np.mean(labels[m] == (probs[m] >= 0.5))
        ece += (np.sum(m) / len(probs)) * np.abs(acc - conf)
    return float(ece)

def temperature_scale(logits: np.ndarray, labels: np.ndarray, init_T: float = 1.0) -> float:
    from scipy.optimize import minimize
    def nll(T):
        T = max(1e-3, T)
        z = logits / T
        e = np.exp(z - z.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        y = labels.astype(int)
        eps = 1e-12
        return -np.mean(np.log(p[np.arange(len(y)), y] + eps))
    res = minimize(lambda t: nll(float(t)), x0=[init_T], method="L-BFGS-B", bounds=[(1e-3, 100.0)])
    return float(res.x[0])

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return {"macro_f1": float(macro_f1), "accuracy": float(acc)}
