from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics import f1_score

from .core import (
    iter_trip_dirs,
    load_trip,
    load_artifact,
    CandidateIndex,
    predict_trip,
    viterbi,
    collapse_route,
    check_route_continuity
)

# Configure logging for Laptop 1 terminal demonstration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("pathsnap.api")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = Path(os.getenv("PATHSNAP_MODEL", str(ROOT / "artifacts" / "pathsnap.joblib")))
DATA_ROOT = Path(os.getenv("PATHSNAP_EVAL_DATA", str(ROOT / "data" / "evaluator_test")))
STATIC_DIR = ROOT / "evaluator_ui"

app = FastAPI(
    title="PathSnap Laptop 1 Map-Matching API",
    description="Distributed map-matching and sequential route reconstruction service for PES University Hackathon.",
    version="2.0.0"
)

MODEL_BUNDLE = None
origins = [x.strip() for x in os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

def get_bundle():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError(f"Model artifact not found at {ARTIFACT}")
        logger.info(f"Loading model artifact from: {ARTIFACT}")
        MODEL_BUNDLE = load_artifact(ARTIFACT)
        model_name = type(MODEL_BUNDLE["model"]).__name__
        logger.info(f"Active inference model loaded successfully: {model_name}")
    return MODEL_BUNDLE

@app.on_event("startup")
def startup_event():
    logger.info("Initializing PathSnap Laptop 1 Server...")
    logger.info(f"Configured Model Artifact : {ARTIFACT.resolve()}")
    logger.info(f"Configured Data Root      : {DATA_ROOT.resolve()} (Exists: {DATA_ROOT.exists()})")
    try:
        get_bundle()
    except Exception as e:
        logger.warning(f"Startup model load warning: {e}")

@app.get("/v1/health")
def health():
    bundle_ok = False
    model_family = None
    try:
        b = get_bundle()
        bundle_ok = b is not None
        model_family = type(b["model"]).__name__
    except Exception:
        pass
    return {
        "status": "ok" if (ARTIFACT.exists() and DATA_ROOT.is_dir()) else "degraded",
        "model_loaded": bundle_ok,
        "model_family": model_family,
        "artifact_path": str(ARTIFACT),
        "data_root_exists": DATA_ROOT.is_dir(),
        "timestamp": time.time()
    }

@app.get("/v1/trips")
def trips():
    try:
        dirs = iter_trip_dirs(DATA_ROOT)
        trip_ids = [p.name for p in dirs]
        logger.info(f"GET /v1/trips -> returned {len(trip_ids)} evaluation trips")
        return {
            "count": len(trip_ids),
            "trip_ids": trip_ids,
            "data_root": str(DATA_ROOT.resolve())
        }
    except FileNotFoundError as e:
        logger.error(f"Error listing trips from {DATA_ROOT}: {e}")
        raise HTTPException(503, str(e))

@app.post("/v1/predict")
def predict(req: PredictRequest):
    req_start = time.perf_counter()
    dirs = {p.name: p for p in iter_trip_dirs(DATA_ROOT)} if DATA_ROOT.exists() else {}
    if req.trip_id not in dirs:
        logger.warning(f"Inference request for unknown trip: {req.trip_id}")
        raise HTTPException(404, f"Trip ID {req.trip_id} not found in evaluation dataset.")

    trip_folder = dirs[req.trip_id]
    logger.info(f">>> INFERENCE REQUEST: Processing trip {req.trip_id}...")

    try:
        bundle = get_bundle()
        # Predictions are generated without exposing ground-truth .route
        trip = load_trip(trip_folder, require_route=False, include_route=False)
    except Exception as e:
        logger.error(f"Error loading trip {req.trip_id}: {e}")
        raise HTTPException(500, f"Error reading trip files: {e}")

    # Pipeline execution
    t_inf_start = time.perf_counter()
    predicted = predict_trip(bundle["model"], trip)
    decoded = viterbi(trip, predicted["candidates"], bundle["model"])
    inf_elapsed_ms = (time.perf_counter() - t_inf_start) * 1000.0

    # Build GeoJSON road coordinates from decoded arc sequence
    unique_decoded_arcs = collapse_route(decoded)
    coords = []
    for aid in unique_decoded_arcs:
        u, v = trip.arcs[int(aid)]
        coords.append([trip.nodes[u].tolist(), trip.nodes[v].tolist()])

    # Sample GPS observations for client-side rendering (limit to 1200 points if huge)
    track_pts = trip.track[:, :2]
    step = max(1, len(track_pts) // 1000)
    gps_sample = track_pts[::step].tolist()

    # Server-Side Evaluation against ground truth (strictly isolated from client prediction)
    metrics = {}
    route_file = trip_folder / f"{req.trip_id}.route"
    if route_file.exists():
        ref = load_trip(trip_folder, require_route=True, include_route=True)
        truth = CandidateIndex(ref).aligned_route()
        
        # Pointwise accuracy
        raw_acc = float((predicted["arc_ids"] == truth).mean())
        dec_acc = float((decoded == truth).mean())
        
        # Macro F1
        macro_f1 = float(f1_score(truth, decoded, average="macro", zero_division=0))
        
        # Candidate miss rate
        misses = sum(int(truth[i] not in predicted["candidates"][i]) for i in range(len(truth)))
        miss_rate = float(misses / max(1, len(truth)))
        
        # Route continuity on collapsed traversed sequence
        raw_route = collapse_route(predicted["arc_ids"])
        dec_route = collapse_route(decoded)
        raw_cont = float(check_route_continuity(raw_route, ref.arcs))
        dec_cont = float(check_route_continuity(dec_route, ref.arcs))
        
        metrics = {
            "classifier_top1_accuracy": round(raw_acc, 4),
            "decoded_top1_accuracy": round(dec_acc, 4),
            "accuracy_gain": round(dec_acc - raw_acc, 4),
            "decoded_macro_f1": round(macro_f1, 4),
            "candidate_miss_rate": round(miss_rate, 4),
            "classifier_route_continuity": round(raw_cont, 4),
            "decoded_route_continuity": round(dec_cont, 4),
            "route_continuity": round(dec_cont, 4),
            "observations_count": len(trip.track),
            "decoded_segments_count": len(dec_route)
        }
        
        logger.info(
            f"<<< TRIP {req.trip_id} EVALUATED: "
            f"Fixes={len(trip.track)} | "
            f"Classifier Top-1={raw_acc*100:.2f}% -> Decoded Top-1={dec_acc*100:.2f}% (+{(dec_acc-raw_acc)*100:.2f}%) | "
            f"Macro-F1={macro_f1*100:.2f}% | Continuity={dec_cont*100:.2f}% | "
            f"Inference={inf_elapsed_ms:.1f}ms"
        )
    else:
        logger.info(f"<<< TRIP {req.trip_id} COMPLETED: No server-side route file. Inference={inf_elapsed_ms:.1f}ms")

    total_request_ms = (time.perf_counter() - req_start) * 1000.0

    return {
        "trip_id": req.trip_id,
        "route_geojson": {
            "type": "MultiLineString",
            "coordinates": coords
        },
        "gps_points": gps_sample,
        "arc_ids": decoded.tolist(),
        "classifier_arc_ids": predicted["arc_ids"].tolist(),
        "metrics": metrics,
        "inference_ms": round(inf_elapsed_ms, 2),
        "total_request_ms": round(total_request_ms, 2)
    }

# Serve static UI files
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
