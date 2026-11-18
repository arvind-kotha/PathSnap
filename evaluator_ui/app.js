// PathSnap Evaluator Dashboard Application Logic
document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const apiUrlInput = document.getElementById('apiUrl');
  const btnConnect = document.getElementById('btnConnect');
  const connectionBadge = document.getElementById('connectionBadge');
  const statusText = document.getElementById('statusText');
  const tripsList = document.getElementById('tripsList');
  const tripCountBadge = document.getElementById('tripCountBadge');
  const tripBanner = document.getElementById('tripBanner');
  const activeTripIdEl = document.getElementById('activeTripId');
  const bannerStats = document.getElementById('bannerStats');
  const btnRunActive = document.getElementById('btnRunActive');
  const btnRunAll = document.getElementById('btnRunAll');
  
  // KPI Elements
  const valDecodedAcc = document.getElementById('valDecodedAcc');
  const subRawAcc = document.getElementById('subRawAcc');
  const valContinuity = document.getElementById('valContinuity');
  const valMacroF1 = document.getElementById('valMacroF1');
  const valMissRate = document.getElementById('valMissRate');
  const valLatency = document.getElementById('valLatency');
  const subFixCount = document.getElementById('subFixCount');

  // Comparison Tab Elements
  const compRawAcc = document.getElementById('compRawAcc');
  const compRawSegments = document.getElementById('compRawSegments');
  const compRawCont = document.getElementById('compRawCont');
  const compDecAcc = document.getElementById('compDecAcc');
  const compDecSegments = document.getElementById('compDecSegments');
  const compDecCont = document.getElementById('compDecCont');
  const jsonViewer = document.getElementById('jsonViewer');
  const benchmarkBody = document.getElementById('benchmarkBody');

  // Map Toggles
  const toggleGps = document.getElementById('toggleGps');
  const toggleRoute = document.getElementById('toggleRoute');
  const btnFitMap = document.getElementById('btnFitMap');

  // State
  let availableTrips = [];
  let selectedTripId = null;
  let currentPredictionData = null;

  // Initialize Leaflet Map
  const map = L.map('map', {
    center: [52.52, 13.405], // Default center, will fit bounds upon load
    zoom: 13,
    zoomControl: true
  });

  // Base tile layer: Esri WorldStreetMap
  const esriTileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012',
    maxZoom: 19
  }).addTo(map);

  // Layer groups for dynamic map elements
  const gpsLayerGroup = L.layerGroup().addTo(map);
  const routeLayerGroup = L.layerGroup().addTo(map);
  const markersLayerGroup = L.layerGroup().addTo(map);

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      button.classList.add('active');
      const targetId = button.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Layer toggles
  toggleGps.addEventListener('change', (e) => {
    if (e.target.checked) map.addLayer(gpsLayerGroup);
    else map.removeLayer(gpsLayerGroup);
  });

  toggleRoute.addEventListener('change', (e) => {
    if (e.target.checked) map.addLayer(routeLayerGroup);
    else map.removeLayer(routeLayerGroup);
  });

  btnFitMap.addEventListener('click', () => {
    fitMapToBounds();
  });

  // Auto-detect server URL from the current window location when loaded over HTTP/HTTPS
  if (window.location && window.location.protocol && window.location.protocol.startsWith('http')) {
    apiUrlInput.value = window.location.origin;
  }

  function getBaseUrl() {
    let url = apiUrlInput.value.trim();
    if (url.endsWith('/')) url = url.slice(0, -1);
    return url;
  }

  // Ping API & Load Trips
  async function checkApiConnection() {
    const baseUrl = getBaseUrl();
    statusText.textContent = 'Connecting...';
    connectionBadge.className = 'badge badge-offline';

    try {
      const resp = await fetch(`${baseUrl}/v1/health`, { method: 'GET' });
      if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
      const data = await resp.json();

      if (data.status === 'ok' || data.model_loaded) {
        connectionBadge.className = 'badge badge-online';
        statusText.textContent = 'Laptop 1 Online';
        loadTrips();
      } else {
        connectionBadge.className = 'badge badge-offline';
        statusText.textContent = 'Model Missing';
      }
    } catch (err) {
      connectionBadge.className = 'badge badge-offline';
      statusText.textContent = 'Offline';
      tripsList.innerHTML = `<div class="error-msg" style="color:#f87171; font-size:0.8rem; padding:10px;">
        Unable to connect to Laptop 1 API at <strong>${baseUrl}</strong>.<br><br>
        Ensure the FastAPI server is running on Laptop 1 with:<br>
        <code>uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000</code>
      </div>`;
    }
  }

  btnConnect.addEventListener('click', checkApiConnection);

  // Load Trips
  async function loadTrips() {
    const baseUrl = getBaseUrl();
    try {
      const resp = await fetch(`${baseUrl}/v1/trips`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      availableTrips = data.trip_ids || [];
      renderTripsList();
      btnRunAll.disabled = availableTrips.length === 0;
    } catch (err) {
      tripsList.innerHTML = `<div style="color:#f87171; font-size:0.75rem;">Failed to fetch trips: ${err.message}</div>`;
    }
  }

  function renderTripsList() {
    tripsList.innerHTML = '';
    tripCountBadge.textContent = `${availableTrips.length} Reserved Trips`;

    availableTrips.forEach((tripId, idx) => {
      const card = document.createElement('div');
      card.className = `trip-card ${selectedTripId === tripId ? 'active' : ''}`;
      card.innerHTML = `
        <div class="trip-info">
          <div class="trip-id-text">${tripId}</div>
          <div class="trip-meta">Evaluation Trip #${idx + 1}</div>
        </div>
        <span class="btn btn-sm btn-secondary">Select</span>
      `;
      card.addEventListener('click', () => selectTrip(tripId));
      tripsList.appendChild(card);
    });

    if (availableTrips.length > 0 && !selectedTripId) {
      selectTrip(availableTrips[0]);
    }
  }

  function selectTrip(tripId) {
    selectedTripId = tripId;
    document.querySelectorAll('.trip-card').forEach(card => {
      const idText = card.querySelector('.trip-id-text')?.textContent;
      card.classList.toggle('active', idText === tripId);
    });

    activeTripIdEl.textContent = `Trip: ${tripId}`;
    bannerStats.textContent = `Ready for server-side map matching. Click "Run Map-Matching" to evaluate.`;
    btnRunActive.disabled = false;
  }

  btnRunActive.addEventListener('click', () => {
    if (selectedTripId) runPrediction(selectedTripId);
  });

  // Run Prediction on Single Trip
  async function runPrediction(tripId) {
    const baseUrl = getBaseUrl();
    btnRunActive.disabled = true;
    btnRunActive.textContent = '⏳ Processing on Laptop 1...';
    bannerStats.textContent = 'Sending request to Laptop 1 backend; running spatial indexing, feature extraction, classifier scoring, and HMM Viterbi decoding...';

    try {
      const t0 = performance.now();
      const resp = await fetch(`${baseUrl}/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trip_id: tripId })
      });
      const clientElapsed = performance.now() - t0;

      if (!resp.ok) {
        const errDetail = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${errDetail}`);
      }

      const data = await resp.json();
      currentPredictionData = data;
      renderPredictionResults(data, clientElapsed);
    } catch (err) {
      alert(`Map matching failed: ${err.message}`);
      bannerStats.textContent = `Error during inference: ${err.message}`;
    } finally {
      btnRunActive.disabled = false;
      btnRunActive.textContent = '▶ Run Map-Matching';
    }
  }

  function renderPredictionResults(data, clientElapsed) {
    const metrics = data.metrics || {};
    const decAcc = metrics.decoded_top1_accuracy !== undefined ? (metrics.decoded_top1_accuracy * 100.0).toFixed(2) : '--';
    const rawAcc = metrics.classifier_top1_accuracy !== undefined ? (metrics.classifier_top1_accuracy * 100.0).toFixed(2) : '--';
    const cont = metrics.decoded_route_continuity !== undefined ? (metrics.decoded_route_continuity * 100.0).toFixed(2) :
                 (metrics.route_continuity !== undefined ? (metrics.route_continuity * 100.0).toFixed(2) : '--');
    const f1 = metrics.decoded_macro_f1 !== undefined ? (metrics.decoded_macro_f1 * 100.0).toFixed(2) : '--';
    const miss = metrics.candidate_miss_rate !== undefined ? (metrics.candidate_miss_rate * 100.0).toFixed(2) : '0.00';
    const latency = data.inference_ms !== undefined ? data.inference_ms.toFixed(1) : clientElapsed.toFixed(1);
    const fixes = data.arc_ids ? data.arc_ids.length : '--';

    // Update Banner
    bannerStats.innerHTML = `<span><strong>${fixes}</strong> GPS fixes processed in <strong>${latency} ms</strong> on Laptop 1. Ground truth evaluated server-side.</span>`;

    // Update KPI cards
    valDecodedAcc.textContent = `${decAcc}%`;
    subRawAcc.textContent = `Raw Classifier: ${rawAcc}%`;
    valContinuity.textContent = `${cont}%`;
    valMacroF1.textContent = `${f1}%`;
    valMissRate.textContent = `${miss}%`;
    valLatency.textContent = `${latency} ms`;
    subFixCount.textContent = `${fixes} observations scored`;

    // Update Comparison Tab
    compRawAcc.textContent = `${rawAcc}%`;
    compDecAcc.textContent = `${decAcc}% (${(parseFloat(decAcc) - parseFloat(rawAcc)) >= 0 ? '+' : ''}${(parseFloat(decAcc) - parseFloat(rawAcc)).toFixed(2)}% gain)`;
    
    // Count collapsed segments
    const rawUnique = collapseArray(data.classifier_arc_ids || []);
    const decUnique = collapseArray(data.arc_ids || []);
    compRawSegments.textContent = `${rawUnique.length} road segments`;
    compDecSegments.textContent = `${decUnique.length} road segments`;

    const rawCont = metrics.classifier_route_continuity !== undefined ? (metrics.classifier_route_continuity * 100.0).toFixed(2) : '--';
    compRawCont.textContent = `${rawCont}%`;
    compDecCont.textContent = `${cont}%`;

    // Update JSON viewer
    jsonViewer.textContent = JSON.stringify(data, null, 2);

    // Update Map Layers
    renderMapLayers(data);
  }

  function collapseArray(arr) {
    const res = [];
    for (let x of arr) {
      if (res.length === 0 || res[res.length - 1] !== x) res.push(x);
    }
    return res;
  }

  function renderMapLayers(data) {
    gpsLayerGroup.clearLayers();
    routeLayerGroup.clearLayers();
    markersLayerGroup.clearLayers();

    const bounds = L.latLngBounds();

    // 1. Render Matched Route (GeoJSON MultiLineString)
    if (data.route_geojson && data.route_geojson.coordinates) {
      const coords = data.route_geojson.coordinates;
      // Coordinates in GeoJSON: [lon, lat] -> Leaflet: [lat, lon]
      const latlngs = [];
      coords.forEach(segment => {
        if (segment.length >= 2) {
          const p1 = [segment[0][1], segment[0][0]];
          const p2 = [segment[1][1], segment[1][0]];
          latlngs.push([p1, p2]);
          bounds.extend(p1);
          bounds.extend(p2);
        }
      });

      const routePolyline = L.polyline(latlngs, {
        color: '#10b981', // vibrant emerald green
        weight: 5,
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round'
      }).addTo(routeLayerGroup);

      routePolyline.bindPopup(`<strong>Reconstructed Vehicle Route</strong><br>Decoded Arc Count: ${data.arc_ids ? data.arc_ids.length : 'N/A'}`);
    }

    // 2. Render GPS Observations
    if (data.gps_points && data.gps_points.length > 0) {
      data.gps_points.forEach((pt, idx) => {
        const lat = pt[1];
        const lon = pt[0];
        const latlng = [lat, lon];
        bounds.extend(latlng);

        const circle = L.circleMarker(latlng, {
          radius: 3.5,
          fillColor: '#3b82f6',
          color: '#1d4ed8',
          weight: 1,
          opacity: 0.8,
          fillOpacity: 0.7
        }).addTo(gpsLayerGroup);

        circle.bindPopup(`<strong>GPS Observation #${idx + 1}</strong><br>Lat: ${lat.toFixed(6)}<br>Lon: ${lon.toFixed(6)}`);
      });

      // Start & End markers
      const startPt = data.gps_points[0];
      const endPt = data.gps_points[data.gps_points.length - 1];

      const startIcon = L.divIcon({
        className: 'custom-pin start-pin',
        html: '<div style="background:#22c55e; width:12px; height:12px; border-radius:50%; border:2px solid white; box-shadow:0 0 6px rgba(0,0,0,0.5);"></div>',
        iconSize: [16, 16],
        iconAnchor: [8, 8]
      });

      const endIcon = L.divIcon({
        className: 'custom-pin end-pin',
        html: '<div style="background:#ef4444; width:12px; height:12px; border-radius:50%; border:2px solid white; box-shadow:0 0 6px rgba(0,0,0,0.5);"></div>',
        iconSize: [16, 16],
        iconAnchor: [8, 8]
      });

      L.marker([startPt[1], startPt[0]], { icon: startIcon }).bindPopup('<strong>Trip Start</strong>').addTo(markersLayerGroup);
      L.marker([endPt[1], endPt[0]], { icon: endIcon }).bindPopup('<strong>Trip Destination</strong>').addTo(markersLayerGroup);
    }

    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 16 });
    }
  }

  function fitMapToBounds() {
    const allLayers = [gpsLayerGroup, routeLayerGroup, markersLayerGroup];
    const bounds = L.latLngBounds();
    allLayers.forEach(lg => {
      lg.eachLayer(layer => {
        if (layer.getBounds) bounds.extend(layer.getBounds());
        else if (layer.getLatLng) bounds.extend(layer.getLatLng());
      });
    });
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [30, 30] });
    }
  }

  // Batch Run All Trips
  btnRunAll.addEventListener('click', async () => {
    if (!availableTrips || availableTrips.length === 0) return;
    if (!confirm(`Run full evaluation pipeline sequentially across all ${availableTrips.length} reserved evaluation trips on Laptop 1?`)) return;

    btnRunAll.disabled = true;
    btnRunAll.textContent = '⏳ Running Benchmark...';
    benchmarkBody.innerHTML = '<tr><td colspan="8" class="text-center">Running benchmark across all trips. Please wait...</td></tr>';

    // Switch to summary tab
    document.querySelector('[data-tab="tabSummary"]').click();

    const baseUrl = getBaseUrl();
    const rows = [];

    for (let i = 0; i < availableTrips.length; i++) {
      const tripId = availableTrips[i];
      bannerStats.textContent = `Benchmarking trip ${i + 1}/${availableTrips.length}: ${tripId}...`;

      try {
        const t0 = performance.now();
        const resp = await fetch(`${baseUrl}/v1/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ trip_id: tripId })
        });
        const elapsed = performance.now() - t0;
        if (resp.ok) {
          const data = await resp.json();
          rows.push({ tripId, data, elapsed });
        }
      } catch (e) {
        console.error(`Failed to benchmark ${tripId}:`, e);
      }
    }

    renderBenchmarkTable(rows);
    btnRunAll.disabled = false;
    btnRunAll.textContent = '⚡ Run Benchmark Across All 10 Trips';
    bannerStats.textContent = `Batch benchmark completed across ${rows.length} trips.`;
  });

  function renderBenchmarkTable(results) {
    if (results.length === 0) {
      benchmarkBody.innerHTML = '<tr><td colspan="8" class="text-center text-danger">No evaluation results collected.</td></tr>';
      return;
    }

    let totalRaw = 0, totalDec = 0, totalF1 = 0, totalCont = 0, totalTime = 0, totalFixes = 0;

    const html = results.map(r => {
      const d = r.data;
      const m = d.metrics || {};
      const fixes = d.arc_ids ? d.arc_ids.length : 0;
      const raw = m.classifier_top1_accuracy ? m.classifier_top1_accuracy * 100 : 0;
      const dec = m.decoded_top1_accuracy ? m.decoded_top1_accuracy * 100 : 0;
      const gain = dec - raw;
      const f1 = m.decoded_macro_f1 ? m.decoded_macro_f1 * 100 : 0;
      const cont = m.decoded_route_continuity ? m.decoded_route_continuity * 100 :
                   (m.route_continuity ? m.route_continuity * 100 : 0);
      const timeMs = d.inference_ms || r.elapsed;

      totalRaw += raw; totalDec += dec; totalF1 += f1; totalCont += cont; totalTime += timeMs; totalFixes += fixes;

      return `<tr>
        <td><strong>${r.tripId}</strong></td>
        <td>${fixes}</td>
        <td>${raw.toFixed(2)}%</td>
        <td class="text-success"><strong>${dec.toFixed(2)}%</strong></td>
        <td class="${gain >= 0 ? 'text-success' : ''}">${gain >= 0 ? '+' : ''}${gain.toFixed(2)}%</td>
        <td>${f1.toFixed(2)}%</td>
        <td>${cont.toFixed(2)}%</td>
        <td>${timeMs.toFixed(1)} ms</td>
      </tr>`;
    }).join('');

    const n = results.length;
    const summaryHtml = `
      <tr style="background-color:rgba(59,130,246,0.15); font-weight:700;">
        <td>MEAN AVERAGE</td>
        <td>${Math.round(totalFixes / n)}</td>
        <td>${(totalRaw / n).toFixed(2)}%</td>
        <td class="text-success">${(totalDec / n).toFixed(2)}%</td>
        <td class="text-success">+${((totalDec - totalRaw) / n).toFixed(2)}%</td>
        <td>${(totalF1 / n).toFixed(2)}%</td>
        <td>${(totalCont / n).toFixed(2)}%</td>
        <td>${(totalTime / n).toFixed(1)} ms</td>
      </tr>
    `;

    benchmarkBody.innerHTML = html + summaryHtml;
  }

  // Auto connect on startup
  checkApiConnection();
});
