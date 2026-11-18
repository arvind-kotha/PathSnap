from __future__ import annotations

import argparse
from pathlib import Path
import time

import folium
from sklearn.metrics import f1_score

from .core import CandidateIndex, iter_trip_dirs, load_artifact, load_trip, predict_trip, viterbi

ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Tiles © Esri — Sources: Esri, HERE, Garmin, USGS"


def continuity(arcs, route_ids):
    if len(route_ids) < 2:
        return 1.0
    seq = arcs[route_ids]
    return float((seq[:-1, 1] == seq[1:, 0]).mean())


def route_paths(trip, route_ids):
    """Convert directed arc IDs to connected Folium lat/lon line parts."""
    paths = []
    for aid in route_ids:
        u, v = trip.arcs[int(aid)]
        a, b = trip.nodes[int(u)], trip.nodes[int(v)]
        start, end = [float(a[1]), float(a[0])], [float(b[1]), float(b[0])]
        if not paths or paths[-1][-1] != start:
            paths.append([start, end])
        else:
            paths[-1].append(end)
    return paths


def make_map(trip, model, output: Path):
    started = time.perf_counter()
    raw = predict_trip(model, trip)
    decoded = viterbi(trip, raw["candidates"], model)
    index = CandidateIndex(trip)
    truth = index.aligned_route() if trip.route is not None else None
    elapsed_ms = (time.perf_counter() - started) * 1000

    gps = [[float(lat), float(lon)] for lon, lat in trip.track[:, :2]]
    center = [float(trip.track[:, 1].mean()), float(trip.track[:, 0].mean())]
    m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles=None, prefer_canvas=True)
    folium.TileLayer(
        tiles=ESRI_TILES,
        attr=ESRI_ATTR,
        name="Esri World Street Map",
        max_zoom=19,
        overlay=False,
        control=True,
    ).add_to(m)

    gps_layer = folium.FeatureGroup(name="Observed GPS fixes", show=True)
    folium.PolyLine(gps, color="blue", weight=2, opacity=0.45, tooltip="GPS trace").add_to(gps_layer)
    marker_step = max(1, len(gps) // 1000)
    for i in range(0, len(gps), marker_step):
        folium.CircleMarker(
            gps[i], radius=3, color="blue", fill=True, fill_color="blue",
            fill_opacity=0.7, weight=1, tooltip="Raw GPS observation",
        ).add_to(gps_layer)
    gps_layer.add_to(m)
    folium.Marker(gps[0], tooltip="Trip start", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    folium.Marker(gps[-1], tooltip="Trip end", icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(m)

    candidate_layer = folium.FeatureGroup(name="Nearby road candidates", show=False)
    relevant = set(int(a) for group in raw["candidates"] for a in group)
    for aid in relevant:
        u, v = trip.arcs[aid]
        a, b = trip.nodes[int(u)], trip.nodes[int(v)]
        folium.PolyLine([[float(a[1]), float(a[0])], [float(b[1]), float(b[0])]], color="#89939e", weight=1, opacity=0.3).add_to(candidate_layer)
    candidate_layer.add_to(m)

    if trip.route is not None:
        folium.PolyLine(
            route_paths(trip, trip.route), color="green", weight=5, opacity=0.85,
            tooltip="Reference road route", name="Reference route",
        ).add_to(folium.FeatureGroup(name="Reference route", show=True).add_to(m))

    folium.PolyLine(
        route_paths(trip, decoded), color="#d83a45", weight=4, opacity=0.85,
        tooltip="Model route (HMM/Viterbi)",
    ).add_to(folium.FeatureGroup(name="Model route (HMM/Viterbi)", show=True).add_to(m))

    metrics = {"trip_id": trip.trip_id, "gps_fixes": int(len(trip.track)), "predicted_fixes": int(len(decoded)), "inference_ms": round(elapsed_ms, 2)}
    if truth is not None:
        metrics.update({
            "classifier_top1_accuracy": float((raw["arc_ids"] == truth).mean()),
            "decoded_top1_accuracy": float((decoded == truth).mean()),
            "decoded_macro_f1": float(f1_score(truth, decoded, average="macro", zero_division=0)),
            "candidate_miss_rate": float(sum(int(truth[i] not in raw["candidates"][i]) for i in range(len(truth))) / max(1, len(truth))),
            "classifier_route_continuity": continuity(trip.arcs, raw["arc_ids"]),
            "decoded_route_continuity": continuity(trip.arcs, decoded),
        })
    metric_text = "<br>".join(
        f"<b>{k.replace('_', ' ').title()}:</b> {v:.3f}" if isinstance(v, float)
        else f"<b>{k.replace('_', ' ').title()}:</b> {v}" for k, v in metrics.items()
    )
    legend = f'''<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:10px 12px;width:200px;border:1px solid #888;border-radius:4px;font-size:12px;box-shadow:0 1px 5px #888"><b>PathSnap · Trip {trip.trip_id}</b><br><span style="color:blue">●</span> GPS &nbsp; <span style="color:green">━</span> reference &nbsp; <span style="color:#d83a45">━</span> predicted<hr style="margin:5px 0">{metric_text}</div>'''
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    south, north = float(trip.track[:, 1].min()), float(trip.track[:, 1].max())
    west, east = float(trip.track[:, 0].min()), float(trip.track[:, 0].max())
    m.fit_bounds([[south, west], [north, east]], padding=[24, 24])
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    return metrics, len(relevant)


def main():
    parser = argparse.ArgumentParser(description="Generate a Folium map for one PathSnap trip")
    parser.add_argument("--data-root", type=Path, required=True, help="Partition directory, such as student_test")
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/pathsnap.joblib"))
    parser.add_argument("--out", type=Path, help="HTML output path; defaults to outputs/pathsnap_<trip-id>.html")
    args = parser.parse_args()
    folder = {p.name: p for p in iter_trip_dirs(args.data_root)}.get(args.trip_id)
    if folder is None:
        parser.error(f"Trip {args.trip_id} not found under {args.data_root}")
    trip = load_trip(folder)
    bundle = load_artifact(args.model)
    output = args.out or Path("outputs") / f"pathsnap_{args.trip_id}.html"
    metrics, candidate_count = make_map(trip, bundle["model"], output)
    print(f"Folium map saved: {output.resolve()}")
    print(f"Nearby road candidates drawn: {candidate_count}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
