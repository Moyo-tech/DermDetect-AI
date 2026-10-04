"""Standard-library numerical contracts; real predictions/outcomes are not read here.

Policy approved in the relayed M2 final-approval reply. Recorded physical IDs
remain the point-estimate units; conservative groups only constrain paired draws.
This module is also copied alone into the inspected CPU development bundle.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math
import random

CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
POLICY = "physical-class-equal-unit-view-loss-and-paired-clusters-v1"
SEED = 20260729
REPLICATES = 1000


@dataclass(frozen=True)
class View:
    image_id: str
    physical_id: str
    cluster_id: str
    label: str
    sha256: str
    active: bool = True


def validated_views(views):
    rows = tuple(views)
    if not rows or any(not isinstance(v, View) for v in rows):
        raise ValueError("Nonempty structured view membership required")
    images, physical, active_hashes = set(), {}, set()
    for v in rows:
        if (any(not isinstance(s, str) or not s or s != s.strip()
                for s in (v.image_id, v.physical_id, v.cluster_id, v.sha256))
                or v.label not in CLASS_ORDER or type(v.active) is not bool):
            raise ValueError("Noncanonical view fields")
        if v.image_id in images:
            raise ValueError("Duplicate image membership")
        images.add(v.image_id)
        key = (v.label, v.cluster_id)
        if v.physical_id in physical and physical[v.physical_id] != key:
            raise ValueError("Physical identity crosses labels or conservative groups")
        physical[v.physical_id] = key
        if v.active:
            pair = (v.physical_id, v.sha256)
            if pair in active_hashes:
                raise ValueError("Repeated exact-content representative within a physical unit")
            active_hashes.add(pair)
    if {v.physical_id for v in rows if v.active} != set(physical):
        raise ValueError("Every physical unit requires an active representative")
    return tuple(sorted(rows, key=lambda v: v.image_id))


def training_weights(views):
    rows = validated_views(views)
    active = tuple(v for v in rows if v.active)
    units = {v.physical_id: v.label for v in active}
    n = Counter(units.values())
    if set(n) != set(CLASS_ORDER):
        raise ValueError("All seven training classes required")
    m = Counter(v.physical_id for v in active)
    total = len(active)
    weights = {v.image_id: Fraction(total, len(CLASS_ORDER) * n[v.label] * m[v.physical_id])
               if v.active else Fraction(0) for v in rows}
    return {"policy": POLICY, "physical_units": len(units), "active_views": total,
            "class_counts": dict(n),
            "class_weights": {c: Fraction(len(units), len(CLASS_ORDER) * n[c]) for c in CLASS_ORDER},
            "image_weights": weights}


def mean_physical_logits(views, image_logits):
    rows = validated_views(views)
    active = tuple(v for v in rows if v.active)
    if set(image_logits) != {v.image_id for v in active}:
        raise ValueError("Exactly one logit vector per active representative required")
    grouped = defaultdict(list)
    for v in active:
        vector = tuple(image_logits[v.image_id])
        if len(vector) != len(CLASS_ORDER) or any(type(x) not in (float, int) or not math.isfinite(x) for x in vector):
            raise ValueError("Finite seven-class logits required")
        grouped[v.physical_id].append(vector)
    return {l: tuple(math.fsum(vector[c] for vector in vectors) / len(vectors)
                     for c in range(len(CLASS_ORDER))) for l, vectors in sorted(grouped.items())}


def equal_physical_nll(physical_logits, labels, temperature):
    if (type(temperature) not in (float, int) or not math.isfinite(temperature)
            or temperature <= 0 or not physical_logits or set(physical_logits) != set(labels)):
        raise ValueError("Positive finite temperature and exact physical membership required")
    losses = []
    for identity in sorted(labels):
        if labels[identity] not in CLASS_ORDER:
            raise ValueError("Unknown physical label")
        vector = tuple(physical_logits[identity])
        if len(vector) != 7 or any(type(x) not in (float, int) or not math.isfinite(x) for x in vector):
            raise ValueError("Finite seven-class logits required")
        scaled = tuple(x / temperature for x in vector)
        if any(not math.isfinite(x) for x in scaled):
            raise ValueError("Temperature caused numeric overflow")
        maximum = max(scaled)
        losses.append(math.log(math.fsum(math.exp(x - maximum) for x in scaled))
                      + maximum - scaled[CLASS_ORDER.index(labels[identity])])
    return math.fsum(losses) / len(losses)


def paired_cluster_draws(views, *, partition, seed=SEED, replicate_count=REPLICATES):
    """Yield ID multiplicities shared by both systems; never merge predictions.

    Class draws count groups, not physical units. No train bootstrap or mixed-label
    evaluation alternative is authorized. Defaults are the approved fixed values;
    explicit alternatives are useful only for invented numerical tests.
    """
    rows = validated_views(views)
    if partition not in ("validation", "internal_test"):
        raise ValueError("Evaluation partition required; training bootstrap is not approved")
    if type(seed) is not int or seed < 0 or type(replicate_count) is not int or replicate_count < 1:
        raise ValueError("Invalid paired draw configuration")
    units = {v.physical_id: (v.cluster_id, v.label) for v in rows}
    clusters = defaultdict(set)
    for identity, (cluster, label) in units.items():
        clusters[cluster].add(identity)
    strata = defaultdict(list)
    for cluster in sorted(clusters):
        labels = {units[l][1] for l in clusters[cluster]}
        if len(labels) != 1:
            raise ValueError("Mixed-label evaluation cluster requires new policy review")
        strata[next(iter(labels))].append(cluster)
    if set(strata) != set(CLASS_ORDER):
        raise ValueError("Every evaluation class must be represented")
    rng = random.Random(seed)
    for _ in range(replicate_count):
        multiplicity = dict.fromkeys(sorted(units), 0)
        selected = []
        for c in CLASS_ORDER:
            bucket = strata[c]
            for _ in bucket:
                cluster = bucket[rng.randrange(len(bucket))]
                selected.append(cluster)
                for identity in clusters[cluster]:
                    multiplicity[identity] += 1
        yield tuple(selected), multiplicity


def paired_indices(multiplicity, ids_a, ids_b):
    """Map one shared ID draw into system-specific row orders."""
    a, b = tuple(ids_a), tuple(ids_b)
    if (len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) != set(b)
            or set(a) != set(multiplicity)
            or any(type(k) is not int or k < 0 for k in multiplicity.values())):
        raise ValueError("Exact paired physical coverage and integer multiplicities required")
    positions_a, positions_b = {v:i for i,v in enumerate(a)}, {v:i for i,v in enumerate(b)}
    repeated = tuple(l for l in sorted(multiplicity) for _ in range(multiplicity[l]))
    return tuple(positions_a[l] for l in repeated), tuple(positions_b[l] for l in repeated)


def percentile_interval(values):
    """Linear empirical quantiles at .025 and .975; no dropped replicates."""
    values = tuple(values)
    if len(values) < 2 or any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        raise ValueError("At least two finite replicate values required")
    ordered = sorted(values)
    def quantile(p):
        position = (len(ordered)-1)*p
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)
    return quantile(.025), quantile(.975)
