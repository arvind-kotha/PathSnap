from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import time

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, log_loss

# Each candidate road segment is scored independently using these eleven
# observations. The same feature order is used for training and inference.
FEATURES = ["distance_m", "heading_delta", "speed_mps", "time_gap_s", "edge_bearing", "edge_length_m", "projection", "candidate_rank", "distance_to_start_m", "distance_to_end_m", "heading_rank"]

@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None = None

def read_table(path: Path, columns: int, dtype=float) -> np.ndarray:
    try:
        a = np.loadtxt(path, dtype=dtype, ndmin=2)
    except ValueError as e:
        raise ValueError(f"Invalid whitespace-separated file {path}: {e}") from e
    if a.shape[1] != columns:
        raise ValueError(f"{path} has {a.shape[1]} columns; expected {columns}")
    return a

def load_trip(folder: Path, require_route: bool = True, include_route: bool = True) -> Trip:
    tid = folder.name
    track = read_table(folder / f"{tid}.track", 3)
    nodes = read_table(folder / f"{tid}.nodes", 2)
    arcs = read_table(folder / f"{tid}.arcs", 2, int).astype(np.int64)
    route_path = folder / f"{tid}.route"
    route = read_table(route_path, 1, int).reshape(-1).astype(np.int64) if include_route and route_path.exists() else None
    if require_route and route is None:
        raise ValueError(f"Missing reference route: {route_path}")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"Non-finite coordinates in trip {tid}")
    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
        raise ValueError(f"Empty track/network in trip {tid}")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Arc endpoint outside node array in trip {tid}")
    if route is not None and (np.any(route < 0) or np.any(route >= len(arcs))):
        raise ValueError(f"Route arc id outside arc array in trip {tid}")
    if len(track) > 1 and np.any(np.diff(track[:, 2]) <= 0):
        raise ValueError(f"Timestamps are not strictly increasing in trip {tid}")
    return Trip(tid, track, nodes, arcs, route)

def metric_coords(lonlat: np.ndarray, origin: np.ndarray) -> np.ndarray:
    # A local equirectangular projection is accurate enough for a compact trip
    # and makes road distances, speeds, and segment lengths available in metres.
    scale_x = 111320.0 * math.cos(math.radians(float(origin[1])))
    scale_y = 110574.0
    return np.column_stack(((lonlat[:, 0] - origin[0]) * scale_x, (lonlat[:, 1] - origin[1]) * scale_y))

def _bearing(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(float(b[0] - a[0]), float(b[1] - a[1]))

class CandidateIndex:
    def __init__(self, trip: Trip):
        self.trip = trip
        self.origin = np.median(trip.track[:, :2], axis=0)
        self.nodes = metric_coords(trip.nodes, self.origin)
        self.points = metric_coords(trip.track[:, :2], self.origin)
        self.starts = self.nodes[trip.arcs[:, 0]]
        self.ends = self.nodes[trip.arcs[:, 1]]
        self.mid = (self.starts + self.ends) / 2
        self.lengths = np.linalg.norm(self.ends - self.starts, axis=1)
        self.bearings = np.arctan2(self.ends[:, 0] - self.starts[:, 0], self.ends[:, 1] - self.starts[:, 1])
        self.node_tree = cKDTree(self.nodes)
        self.mid_tree = cKDTree(self.mid)
        self.incident: list[list[int]] = [[] for _ in range(len(self.nodes))]
        for aid, (u, v) in enumerate(trip.arcs):
            self.incident[int(u)].append(aid)
            self.incident[int(v)].append(aid)

    def candidates(self, point: np.ndarray, k: int = 32, radius: float = 250.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Query both segment endpoints and midpoints, then keep the closest
        # unique road segments. This keeps candidate generation fast per GPS fix.
        kn = min(k, len(self.nodes)); ke = min(k, len(self.mid))
        ni = np.atleast_1d(self.node_tree.query(point, k=kn)[1]).astype(int)
        mi = np.atleast_1d(self.mid_tree.query(point, k=ke)[1]).astype(int)
        ids = set(int(x) for x in mi)
        for n in ni:
            ids.update(self.incident[int(n)])
        if not ids:
            ids.add(int(self.mid_tree.query(point, k=1)[1]))
        arr = np.fromiter(ids, dtype=np.int64)
        seg = self.ends[arr] - self.starts[arr]
        denom = np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-8)
        frac = np.clip(np.einsum("ij,ij->i", point - self.starts[arr], seg) / denom, 0, 1)
        nearest = self.starts[arr] + frac[:, None] * seg
        dist = np.linalg.norm(point - nearest, axis=1)
        order = np.argsort(dist)
        arr, frac, dist = arr[order], frac[order], dist[order]
        keep = np.flatnonzero(dist <= radius)[:12]
        if len(keep) == 0:
            keep = np.array([0])
        return arr[keep], frac[keep], dist[keep]

    def aligned_route(self) -> np.ndarray:
        """Closest ordered reference-route arc per GPS fix; route is used only for labels/scoring."""
        if self.trip.route is None:
            raise ValueError("Reference route required for alignment")
        route = self.trip.route
        mids=self.mid[route]; tree=cKDTree(mids); k=min(32,len(route))
        _, near=tree.query(self.points,k=k); near=np.asarray(near).reshape(len(self.points),k)
        labels=np.empty(len(self.points),dtype=np.int64)
        for i,(point,choices) in enumerate(zip(self.points,near)):
            ids=route[choices]; seg=self.ends[ids]-self.starts[ids]
            den=np.maximum(np.einsum('ij,ij->i',seg,seg),1e-8)
            frac=np.clip(np.einsum('ij,ij->i',point-self.starts[ids],seg)/den,0,1)
            q=self.starts[ids]+frac[:,None]*seg
            labels[i]=int(ids[np.argmin(np.linalg.norm(point-q,axis=1))])
        return labels

    def feature_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        labels = self.aligned_route()
        xrows=[]; yrows=[]; groups=[]; misses=0
        track=self.trip.track
        for i, p in enumerate(self.points):
            ids, frac, dist = self.candidates(p)
            if labels[i] not in ids: misses += 1
            dt = max(float(track[i,2] - track[i-1,2]), 1e-3) if i else 1.0
            bearing = _bearing(self.points[i-1], p) if i else float(self.bearings[ids[0]])
            speed = float(np.linalg.norm(p-self.points[i-1])/dt) if i else 0.0
            delta = np.arctan2(np.sin(self.bearings[ids]-bearing), np.cos(self.bearings[ids]-bearing))
            ranks=np.arange(len(ids),dtype=float)
            heading_delta = np.abs(delta)
            # Endpoint distances and relative heading rank distinguish roads
            # that have similar perpendicular GPS distance.
            start_dist = np.linalg.norm(p-self.starts[ids], axis=1)
            end_dist = np.linalg.norm(p-self.ends[ids], axis=1)
            heading_rank = np.argsort(np.argsort(heading_delta)).astype(float)
            rows=np.column_stack((dist, heading_delta, np.full(len(ids),speed), np.full(len(ids),dt), self.bearings[ids], self.lengths[ids], frac, ranks, start_dist, end_dist, heading_rank))
            xrows.append(rows); yrows.append((ids == labels[i]).astype(np.int8)); groups.append(np.full(len(ids), i, dtype=np.int64))
        return np.vstack(xrows), np.concatenate(yrows), np.concatenate(groups), misses

def iter_trip_dirs(root: Path) -> list[Path]:
    if not root.is_dir(): raise FileNotFoundError(f"Dataset partition not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())

def build_dataset(root: Path) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    xs=[]; ys=[]; gs=[]; trip_groups=[]; miss=[]; offset=0
    dirs=iter_trip_dirs(root)
    for ti, folder in enumerate(dirs):
        trip=load_trip(folder); index=CandidateIndex(trip)
        x,y,g,m=index.feature_rows(); xs.append(x); ys.append(y); gs.append(g+offset); trip_groups.append(np.full(len(y),ti)); miss.append(m/max(1,len(trip.track))); offset += len(trip.track)
        print(f"prepared {trip.trip_id}: fixes={len(trip.track)} candidates={len(y)} candidate_miss={miss[-1]:.3f}")
    return np.vstack(xs).astype(np.float32),np.concatenate(ys),np.concatenate(gs),np.concatenate(trip_groups),np.asarray(miss)

def new_model(name: str, seed: int = 42, params: dict[str,Any] | None = None):
    p=params or {}
    if name=="logistic": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=p.get("C",1.0), random_state=seed))
    if name=="naive_bayes": return GaussianNB(var_smoothing=p.get("var_smoothing",1e-9))
    if name=="random_forest": return RandomForestClassifier(n_estimators=p.get("n_estimators",150), max_depth=p.get("max_depth",18), min_samples_leaf=p.get("min_samples_leaf",2), class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name=="hist_gradient": return HistGradientBoostingClassifier(
        max_iter=p.get("max_iter",150),
        learning_rate=p.get("learning_rate",0.08),
        max_leaf_nodes=p.get("max_leaf_nodes",15),
        l2_regularization=p.get("l2_regularization",0.1),
        random_state=seed,
    )
    if name=="mlp": return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=p.get("hidden_layer_sizes",(32,)),
        activation=p.get("activation","relu"),
        solver=p.get("solver","adam"),
        alpha=p.get("alpha",0.0005),
        learning_rate_init=p.get("learning_rate_init",0.001),
        max_iter=p.get("max_iter",250),
        early_stopping=True,
        validation_fraction=p.get("validation_fraction",0.15),
        n_iter_no_change=p.get("n_iter_no_change",15),
        random_state=seed,
    ))
    if name=="extra_trees": return ExtraTreesClassifier(n_estimators=160,max_depth=18,class_weight="balanced",n_jobs=-1,random_state=seed)
    raise ValueError(f"Unknown model: {name}")

def fit_model(model, x, y):
    # Preserve all positives up to the cap, then deterministically downsample
    # negatives. This controls memory without changing the prediction pipeline.
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    rng=np.random.default_rng(42)
    if len(pos)>60000: pos=rng.choice(pos,size=60000,replace=False)
    neg_cap=min(len(neg),len(pos)*4)
    if len(neg)>neg_cap: neg=rng.choice(neg,size=neg_cap,replace=False)
    ids=np.concatenate((pos,neg)); rng.shuffle(ids)
    model.fit(x[ids],y[ids]); return model

def score_rows(model, x, y, groups):
    prob=model.predict_proba(x)[:,1]
    pred=(prob>=0.5).astype(int)
    # candidate-pair metrics plus observation-level top candidate hit rate
    _, inv=np.unique(groups,return_inverse=True); best=np.full(inv.max()+1,-np.inf); chosen=np.zeros(inv.max()+1,dtype=int)
    np.maximum.at(best,inv,prob); selected=np.flatnonzero(prob==best[inv]); chosen[inv[selected]]=y[selected]
    return {"pair_macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),"pair_log_loss":float(log_loss(y,np.column_stack((1-prob,prob)),labels=[0,1])),"observation_top1":float(chosen.mean())}

def predict_trip(model, trip: Trip) -> dict[str,Any]:
    idx=CandidateIndex(trip); cand=[]; feature_blocks=[]
    for i,p in enumerate(idx.points):
        ids,frac,dist=idx.candidates(p); dt=max(float(trip.track[i,2]-trip.track[i-1,2]),1e-3) if i else 1.0
        bearing=_bearing(idx.points[i-1],p) if i else float(idx.bearings[ids[0]])
        speed=float(np.linalg.norm(p-idx.points[i-1])/dt) if i else 0.0
        delta=np.abs(np.arctan2(np.sin(idx.bearings[ids]-bearing),np.cos(idx.bearings[ids]-bearing)))
        start_dist=np.linalg.norm(p-idx.starts[ids],axis=1)
        end_dist=np.linalg.norm(p-idx.ends[ids],axis=1)
        heading_rank=np.argsort(np.argsort(delta)).astype(float)
        features=np.column_stack((dist,delta,np.full(len(ids),speed),np.full(len(ids),dt),idx.bearings[ids],idx.lengths[ids],frac,np.arange(len(ids)),start_dist,end_dist,heading_rank)).astype(np.float32)
        feature_blocks.append(features); cand.append(ids)
    all_probs=model.predict_proba(np.vstack(feature_blocks))[:,1]
    out=[]; offset=0
    for ids,features in zip(cand,feature_blocks):
        probs=all_probs[offset:offset+len(ids)]; offset+=len(ids)
        out.append(int(ids[int(np.argmax(probs))]))
    return {"arc_ids":np.asarray(out,dtype=np.int64),"candidates":cand,"index":idx}

def collapse_route(seq: np.ndarray | list[int]) -> np.ndarray:
    """Compress consecutive identical arc IDs into an ordered sequence of traversed road segments."""
    r = []
    for a in seq:
        aid = int(a)
        if not r or r[-1] != aid:
            r.append(aid)
    return np.asarray(r, dtype=np.int64)

def check_route_continuity(route_arcs: np.ndarray | list[int], arcs_array: np.ndarray) -> float:
    """Calculate the percentage of consecutive road segments that are topologically connected."""
    if len(route_arcs) <= 1:
        return 1.0
    connected = sum(int(arcs_array[int(route_arcs[i]), 1]) == int(arcs_array[int(route_arcs[i+1]), 0])
                    for i in range(len(route_arcs) - 1))
    return float(connected / (len(route_arcs) - 1))

def viterbi(trip: Trip, candidates: list[np.ndarray], model, max_gap: float = 30.0,
            p_stay: float = 0.60, p_direct: float = 0.38, p_jump: float = 1e-5) -> np.ndarray:
    """Decode a connected directed-road sequence; restart after long GPS gaps."""
    arcs = trip.arcs
    n = len(candidates)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    probs = []
    feature_blocks = []
    idx = CandidateIndex(trip)
    for i, ids in enumerate(candidates):
        p = idx.points[i]
        dt = max(float(trip.track[i, 2] - trip.track[i-1, 2]), 1e-3) if i else 1.0
        bearing = _bearing(idx.points[i-1], p) if i else float(idx.bearings[ids[0]])
        speed = float(np.linalg.norm(p - idx.points[i-1]) / dt) if i else 0.0
        frac = np.zeros(len(ids))
        seg = idx.ends[ids] - idx.starts[ids]
        d = np.maximum(np.einsum('ij,ij->i', seg, seg), 1e-8)
        frac = np.clip(np.einsum('ij,ij->i', p - idx.starts[ids], seg) / d, 0, 1)
        dist = np.linalg.norm(p - (idx.starts[ids] + frac[:, None] * seg), axis=1)
        delta = np.abs(np.arctan2(np.sin(idx.bearings[ids] - bearing), np.cos(idx.bearings[ids] - bearing)))
        start_dist = np.linalg.norm(p - idx.starts[ids], axis=1)
        end_dist = np.linalg.norm(p - idx.ends[ids], axis=1)
        heading_rank = np.argsort(np.argsort(delta)).astype(float)
        f = np.column_stack((dist, delta, np.full(len(ids), speed), np.full(len(ids), dt),
                             idx.bearings[ids], idx.lengths[ids], frac, np.arange(len(ids)),
                             start_dist, end_dist, heading_rank)).astype(np.float32)
        feature_blocks.append(f)

    all_probs = np.log(np.maximum(model.predict_proba(np.vstack(feature_blocks))[:, 1], 1e-8))
    offset = 0
    for f in feature_blocks:
        probs.append(all_probs[offset:offset+len(f)])
        offset += len(f)

    # Viterbi recurrence with topological transition probabilities
    scores = probs[0].copy()
    backs = []
    log_stay = math.log(p_stay)
    log_direct = math.log(p_direct)
    log_jump = math.log(p_jump)

    for t in range(1, n):
        prev, cur = candidates[t-1], candidates[t]
        dt = float(trip.track[t, 2] - trip.track[t-1, 2])
        if dt > max_gap:
            backs.append(None)
            scores = probs[t].copy()
            continue

        mat = np.full((len(prev), len(cur)), log_jump)
        for a, pa in enumerate(prev):
            end_node = int(arcs[pa, 1])
            start_node = int(arcs[pa, 0])
            for b, cb in enumerate(cur):
                if pa == cb:
                    mat[a, b] = log_stay
                elif int(arcs[cb, 0]) == end_node:
                    mat[a, b] = log_direct
                elif int(arcs[cb, 0]) == start_node or int(arcs[cb, 1]) == end_node:
                    mat[a, b] = log_direct - 2.5
                else:
                    mat[a, b] = log_jump

        values = scores[:, None] + mat
        back = np.argmax(values, axis=0)
        scores = values[back, np.arange(len(cur))] + probs[t]
        backs.append(back)

    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(scores))
    for t in range(n-1, 0, -1):
        if backs[t-1] is None:
            states[t-1] = int(np.argmax(probs[t-1]))
        else:
            states[t-1] = int(backs[t-1][states[t]])

    return np.asarray([candidates[t][states[t]] for t in range(n)], dtype=np.int64)

def save_artifact(model, path: Path, metadata: dict[str,Any]):
    path.parent.mkdir(parents=True,exist_ok=True); joblib.dump({"model":model,"metadata":metadata},path)

def load_artifact(path: Path):
    if not path.exists(): raise FileNotFoundError(f"Model artifact missing: {path}; run pathsnap-train first")
    return joblib.load(path)
