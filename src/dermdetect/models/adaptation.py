"""CPU probe/calibration contracts; binary policies require explicit caller configuration."""
from collections import Counter
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def checked(logits, labels=None):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Nonempty finite logits required")
    if labels is None:
        return x
    y = np.asarray(labels)
    if y.shape != (len(x),) or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= x.shape[1]):
        raise ValueError("Aligned integer labels required")
    return x, y


def aggregate(ids, physical_ids, logits):
    x = checked(logits)
    if len(ids) != len(x) or len(set(ids)) != len(ids) or len(physical_ids) != len(ids):
        raise ValueError("Exact image/physical alignment required")
    units = tuple(sorted(set(physical_ids)))
    return units, np.stack([x[np.array(physical_ids) == unit].mean(axis=0) for unit in units])


def physical_sample_weights(labels, physical_ids):
    """Balanced physical classes, equal unit contribution divided among views.

    Mean-one weights preserve the solver's total-weight regularization convention.
    Seven-class use implements M2; binary formula remains a review proposal.
    """
    if len(labels) != len(physical_ids) or not labels:
        raise ValueError("Aligned training membership required")
    owners = {}
    for label, unit in zip(labels, physical_ids):
        if unit in owners and owners[unit] != label:
            raise ValueError("Mixed-label physical identity")
        owners[unit] = label
    counts, views = Counter(owners.values()), Counter(physical_ids)
    return np.array([len(labels) / (len(counts) * counts[y] * views[u])
                     for y, u in zip(labels, physical_ids)], dtype=np.float64)


class Probe:
    def __init__(self, c, classes):
        if c <= 0 or tuple(classes) != tuple(range(len(classes))) or len(classes) < 2:
            raise ValueError("Positive C and canonical numeric class order required")
        self.classes = tuple(classes)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=20260729)

    def fit(self, train_features, labels, physical_ids, *, partition):
        if partition != "train" or set(labels) != set(self.classes):
            raise ValueError("Train-only fit with every configured class required")
        x = np.asarray(train_features, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(labels) or not np.isfinite(x).all():
            raise ValueError("Finite aligned train features required")
        # Scaling fitted on train rows only; weighting enters classifier objective once.
        z = self.scaler.fit_transform(x)
        self.model.fit(z, labels, sample_weight=physical_sample_weights(labels, physical_ids))
        if tuple(self.model.classes_) != self.classes:
            raise ValueError("Probe class order mismatch")
        return self

    def logits(self, features):
        score = self.model.decision_function(self.scaler.transform(features))
        return np.column_stack((np.zeros(len(score)), score)) if score.ndim == 1 else score


def selection_key(physical_logits, labels, *, epoch=None, c=None):
    x, y = checked(physical_logits, labels)
    f1 = f1_score(y, x.argmax(axis=1), labels=range(x.shape[1]), average="macro", zero_division=0)
    nll = float(np.mean(logsumexp(x, axis=1) - x[np.arange(len(y)), y]))
    if (epoch is None) == (c is None):
        raise ValueError("Supply epoch or C tie-break, exclusively")
    return (-float(f1), nll, epoch if epoch is not None else c)


def fit_temperature(physical_logits, labels, *, partition):
    if partition != "validation":
        raise ValueError("Validation-only calibration")
    x, y = checked(physical_logits, labels)
    def loss(log_t):
        z = x / math.exp(log_t)
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))
    # Finite numerical bounds are proposed for review, never silently approved.
    result = minimize_scalar(loss, bounds=(-5, 5), method="bounded", options={"xatol": 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Temperature optimization failed")
    return dict(temperature=math.exp(result.x), nll_before=loss(0), nll_after=result.fun,
                at_bound=abs(result.x) > 4.999, numerical_bounds_log_temperature=[-5, 5])


def confidence_cutoff(physical_logits, temperature, coverage, *, partition, tie_policy):
    if partition != "validation" or tie_policy != "include_all_ties" or not 0 < coverage <= 1:
        raise ValueError("Explicit validation-only coverage/tie policy required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Positive temperature required")
    confidence = softmax(checked(physical_logits) / temperature, axis=1).max(axis=1)
    cutoff = float(np.sort(confidence)[len(confidence) - math.ceil(coverage * len(confidence))])
    return dict(cutoff=cutoff, realized_validation_coverage=float(np.mean(confidence >= cutoff)),
                target_coverage=coverage, tie_policy=tie_policy)


def binary_labels(labels, mapping):
    if not mapping or set(mapping.values()) != {0, 1, None}:
        raise ValueError("Explicit benign/malignant/exclusion mapping required")
    if any(label not in mapping for label in labels):
        raise ValueError("Unmapped source label")
    indices = [i for i, label in enumerate(labels) if mapping[label] is not None]
    return indices, [mapping[labels[i]] for i in indices]


def save_probe(directory, probe, configuration):
    """Portable arrays and JSON only; no executable pickle deserialization."""
    import hashlib
    import json
    from pathlib import Path
    directory = Path(directory); directory.mkdir()
    arrays = directory / "probe.npz"
    np.savez(arrays, mean=probe.scaler.mean_, scale=probe.scaler.scale_, variance=probe.scaler.var_,
             coefficients=probe.model.coef_, intercept=probe.model.intercept_, classes=probe.model.classes_)
    metadata = {"configuration":configuration, "C":probe.model.C,
                "classes":probe.classes, "sha256":hashlib.sha256(arrays.read_bytes()).hexdigest()}
    path = directory / "probe.json"
    path.write_text(json.dumps(metadata,sort_keys=True,allow_nan=False)+"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe(directory, expected_configuration, expected_metadata_sha256):
    import hashlib
    import json
    from pathlib import Path
    directory = Path(directory); path = directory / "probe.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_metadata_sha256:
        raise ValueError("Probe metadata changed")
    metadata = json.loads(path.read_text()); arrays = directory / "probe.npz"
    if metadata["configuration"] != expected_configuration or hashlib.sha256(arrays.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Probe configuration/payload changed")
    probe = Probe(metadata["C"],metadata["classes"])
    with np.load(arrays,allow_pickle=False) as data:
        if any(not np.isfinite(data[key]).all() for key in data.files):
            raise ValueError("Nonfinite probe state")
        probe.scaler.mean_ = data["mean"].copy(); probe.scaler.scale_ = data["scale"].copy()
        probe.scaler.var_ = data["variance"].copy(); probe.scaler.n_features_in_ = len(data["mean"])
        probe.model.coef_ = data["coefficients"].copy(); probe.model.intercept_ = data["intercept"].copy()
        probe.model.classes_ = data["classes"].copy(); probe.model.n_features_in_ = len(data["mean"])
    if tuple(probe.model.classes_) != probe.classes or np.any(probe.scaler.scale_ <= 0):
        raise ValueError("Invalid scaler/class order")
    return probe
