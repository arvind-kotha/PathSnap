import sys
import os
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, r"C:\Users\Arvind\PathSnap\src")
from pathsnap.core import load_trip, load_artifact, CandidateIndex, predict_trip, viterbi, collapse_route, check_route_continuity

bundle_path = Path(r"C:\Users\Arvind\PathSnap\artifacts\notebook_models\pathsnap.joblib")
bundle = load_artifact(bundle_path)
model = bundle["model"]

print(f"Loaded model: {type(model).__name__}")

def evaluate_partition(partition_name, folder_path):
    dirs = sorted([d for d in Path(folder_path).iterdir() if d.is_dir() and (d / f"{d.name}.track").exists()])
    print(f"\n==================== Evaluating {partition_name} ({len(dirs)} trips) ====================")
    results = []
    
    t_start = time.perf_counter()
    for d in dirs:
        trip = load_trip(d, require_route=True, include_route=True)
        t0 = time.perf_counter()
        truth = CandidateIndex(trip).aligned_route()
        pred = predict_trip(model, trip)
        decoded = viterbi(trip, pred["candidates"], model)
        elapsed = (time.perf_counter() - t0) * 1000.0
        
        raw_arcs = pred["arc_ids"]
        raw_top1 = float((raw_arcs == truth).mean())
        dec_top1 = float((decoded == truth).mean())
        macro_f1 = float(f1_score(truth, decoded, average="macro", zero_division=0))
        
        raw_route = collapse_route(raw_arcs)
        dec_route = collapse_route(decoded)
        gt_route = collapse_route(truth)
        
        raw_cont = check_route_continuity(raw_route, trip.arcs)
        dec_cont = check_route_continuity(dec_route, trip.arcs)
        gt_cont = check_route_continuity(gt_route, trip.arcs)
        
        misses = sum(int(truth[i] not in pred["candidates"][i]) for i in range(len(truth)))
        miss_rate = misses / max(1, len(truth))
        
        res = {
            "trip_id": trip.trip_id,
            "fixes": len(trip.track),
            "raw_top1_acc": raw_top1 * 100.0,
            "decoded_top1_acc": dec_top1 * 100.0,
            "macro_f1": macro_f1 * 100.0,
            "raw_route_continuity": raw_cont * 100.0,
            "decoded_route_continuity": dec_cont * 100.0,
            "gt_route_continuity": gt_cont * 100.0,
            "candidate_miss_rate": miss_rate * 100.0,
            "latency_ms": elapsed
        }
        results.append(res)
        print(f"[{trip.trip_id}] Fixes: {len(trip.track):4d} | Raw: {res['raw_top1_acc']:.2f}% -> Decoded: {res['decoded_top1_acc']:.2f}% | F1: {res['macro_f1']:.2f}% | Continuity: {res['decoded_route_continuity']:.2f}% | Miss: {res['candidate_miss_rate']:.2f}% | Time: {elapsed:.1f}ms")

    total_time = time.perf_counter() - t_start
    mean_raw = np.mean([r["raw_top1_acc"] for r in results])
    mean_dec = np.mean([r["decoded_top1_acc"] for r in results])
    mean_f1 = np.mean([r["macro_f1"] for r in results])
    mean_cont = np.mean([r["decoded_route_continuity"] for r in results])
    mean_miss = np.mean([r["candidate_miss_rate"] for r in results])
    mean_lat = np.mean([r["latency_ms"] for r in results])
    
    print(f"\n--- {partition_name} SUMMARY ---")
    print(f"Mean Raw Top-1 Accuracy    : {mean_raw:.2f}%")
    print(f"Mean Decoded Top-1 Accuracy: {mean_dec:.2f}% (Gain: +{mean_dec - mean_raw:.2f}%)")
    print(f"Mean Macro F1-Score        : {mean_f1:.2f}%")
    print(f"Mean Decoded Route Cont.   : {mean_cont:.2f}%")
    print(f"Mean Candidate Miss Rate   : {mean_miss:.2f}%")
    print(f"Mean Inference Latency     : {mean_lat:.1f} ms/trip")
    print(f"Total Partition Time       : {total_time:.2f} s")
    
    return results

student_test_dir = r"C:\Users\Arvind\Downloads\ml hack\Student folder\Student folder\student_test"
evaluator_test_dir = r"C:\Users\Arvind\Downloads\ml hack\evaluator_test"

res_student = evaluate_partition("Student Test Partition", student_test_dir)
res_evaluator = evaluate_partition("Evaluator Test Partition", evaluator_test_dir)

# Save results to json
with open(r"C:\Users\Arvind\PathSnap\artifacts\verified_evaluation_results.json", "w") as f:
    json.dump({"student_test": res_student, "evaluator_test": res_evaluator}, f, indent=2)
print("\nSaved verified evaluation results to artifacts/verified_evaluation_results.json")
