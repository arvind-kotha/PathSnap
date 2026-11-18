import sys
import json
from pathlib import Path

sys.path.insert(0, r"C:\Users\Arvind\PathSnap\src")
from fastapi.testclient import TestClient
from pathsnap.api import app

client = TestClient(app)

print("=== 1. Testing GET / ===")
r_root = client.get("/")
print(f"Status: {r_root.status_code}, Content-Type: {r_root.headers.get('content-type')}")
assert r_root.status_code == 200
assert "PathSnap" in r_root.text

print("\n=== 2. Testing GET /v1/health ===")
r_health = client.get("/v1/health")
print(f"Status: {r_health.status_code}")
print(json.dumps(r_health.json(), indent=2))
assert r_health.status_code == 200
assert r_health.json()["status"] == "ok"

print("\n=== 3. Testing GET /v1/trips ===")
r_trips = client.get("/v1/trips")
print(f"Status: {r_trips.status_code}")
trips_data = r_trips.json()
print(f"Trips count: {trips_data['count']}")
print(f"Trip IDs: {trips_data['trip_ids']}")
assert r_trips.status_code == 200
assert len(trips_data["trip_ids"]) == 10

print("\n=== 4. Testing POST /v1/predict for all 10 trips ===")
all_results = []
for tid in trips_data["trip_ids"]:
    r_pred = client.post("/v1/predict", json={"trip_id": tid})
    assert r_pred.status_code == 200, f"Failed for {tid}: {r_pred.text}"
    p_data = r_pred.json()
    m = p_data["metrics"]
    print(f"[{tid}] Decoded Acc: {m['decoded_top1_accuracy']*100:.2f}% | Continuity: {m['decoded_route_continuity']*100:.2f}% | F1: {m['decoded_macro_f1']*100:.2f}% | Inference: {p_data['inference_ms']}ms")
    all_results.append({
        "trip_id": tid,
        "metrics": m,
        "inference_ms": p_data["inference_ms"],
        "route_segments": len(p_data["route_geojson"]["coordinates"]),
        "gps_points_count": len(p_data["gps_points"])
    })

print("\nAll 10 evaluator trips tested successfully through the API!")
with open(r"C:\Users\Arvind\PathSnap\artifacts\api_test_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
