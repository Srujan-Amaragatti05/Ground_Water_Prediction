/**
 * frontend/phase2.js
 * ==================
 * Phase-2 Deep Learning Controller:
 * 
 * 1. EXISTING MONITORING STATIONS
 *    - State -> District -> Monitoring Station -> Prediction Year -> Predict
 *    - Automatic internal 5-year sequence retrieval
 *    - Dynamically populated prediction years based on actual historical observations
 *    - Executes LSTM, GRU, and Hybrid Spatio-Temporal GRU in parallel (no model selector)
 *    - Displays station metadata, prediction results, and collapsible historical input data
 * 
 * 2. NEW LOCATION / NEW WELL PREDICTION
 *    - Latitude, Longitude inputs
 *    - 5 consecutive historical years [WL, Rainfall, Temperature, Humidity]
 *    - Validates consecutive years and matching target prediction year (Year 5 + 1)
 *    - Executes LSTM, GRU, and Hybrid in parallel (no model selector)
 *    - Displays consistent prediction results
 */

(function () {
    "use strict";

    const API_BASE = window.location.origin.includes(':8000') 
        ? window.location.origin 
        : 'http://localhost:8000';

    let phase2Map = null;
    let clusterGroup = null;
    let allStations = [];
    let stationMarkerMap = new Map();
    let currentSelectedStation = null;
    let selectedHighlightMarker = null;

    // Known test station constant
    const TEST_STATION_ID = "West Bengal_Purulia_Santuri_Leadson_23.51992_86.82893";

    // ---------------------------------------------------------------------------
    // Risk categorization utility
    // ---------------------------------------------------------------------------
    function getRiskCategory(wl) {
        if (wl < 3.0) {
            return { label: "Safe", className: "risk-safe" };
        } else if (wl <= 6.0) {
            return { label: "Warning", className: "risk-warning" };
        } else {
            return { label: "Critical", className: "risk-critical" };
        }
    }

    function getDepthColor(wl) {
        if (wl <= 3.0) return "#22c55e"; // green/shallow
        if (wl <= 6.0) return "#eab308"; // yellow/moderate
        return "#ef4444";               // red/deep
    }

    // ---------------------------------------------------------------------------
    // Map Initialization
    // ---------------------------------------------------------------------------
    function initMap() {
        const mapContainer = document.getElementById("phase2-map");
        if (!mapContainer || phase2Map) return;

        phase2Map = L.map("phase2-map", { preferCanvas: true })
            .setView([22.5, 80.0], 5);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "© OpenStreetMap contributors | CGWB Groundwater Monitoring"
        }).addTo(phase2Map);

        clusterGroup = L.markerClusterGroup({
            disableClusteringAtZoom: 14,
            showCoverageOnHover: false,
            maxClusterRadius: 50
        });
        phase2Map.addLayer(clusterGroup);

        const legend = L.control({ position: "bottomright" });
        legend.onAdd = function () {
            const div = L.DomUtil.create("div", "leaflet-legend");
            div.innerHTML = `
                <b>Water Depth (mbgl)</b>
                <div class="legend-item"><span class="legend-color-dot" style="background:#22c55e"></span> Shallow (&lt; 3m)</div>
                <div class="legend-item"><span class="legend-color-dot" style="background:#eab308"></span> Moderate (3–6m)</div>
                <div class="legend-item"><span class="legend-color-dot" style="background:#ef4444"></span> Deep (&gt; 6m)</div>
            `;
            return div;
        };
        legend.addTo(phase2Map);

        window.invalidatePhase2Map = function () {
            if (phase2Map) {
                setTimeout(() => { phase2Map.invalidateSize(); }, 150);
            }
        };
    }

    // ---------------------------------------------------------------------------
    // Load Station Data
    // ---------------------------------------------------------------------------
    async function loadStations() {
        try {
            const res = await fetch("phase2_locations.json");
            if (!res.ok) throw new Error("Failed to load phase2_locations.json");
            allStations = await res.json();
            
            populateStateFilter();
            renderMapMarkers(allStations);

            // Default to known test station if present
            const testStation = allStations.find(s => s.id === TEST_STATION_ID);
            if (testStation) {
                selectStation(testStation, false);
            } else if (allStations.length > 0) {
                selectStation(allStations[0], false);
            }
        } catch (err) {
            console.error("Error loading Phase 2 stations:", err);
            const statusEl = document.getElementById("p2-station-status");
            if (statusEl) {
                statusEl.innerHTML = `<div class="alert alert-error">Failed to load All-India monitoring stations data.</div>`;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Map Marker Rendering
    // ---------------------------------------------------------------------------
    function renderMapMarkers(stations) {
        if (!clusterGroup) return;
        clusterGroup.clearLayers();
        stationMarkerMap.clear();

        stations.forEach(station => {
            const markerColor = getDepthColor(station.recent_wl);
            const marker = L.circleMarker([station.lat, station.lng], {
                radius: station.featured ? 8 : 6,
                weight: station.featured ? 2 : 1,
                color: station.featured ? "#1e3a8a" : "#475569",
                fillColor: markerColor,
                fillOpacity: 0.8
            });

            marker.bindTooltip(`
                <strong>${station.village}</strong> (${station.district}, ${station.state})<br>
                Latest WL: ${station.recent_wl} mbgl<br>
                Available Pred Years: ${station.available_years ? station.available_years.join(', ') : station.pred}
            `);

            marker.on("click", () => {
                // Sync dropdowns
                const stateSelect = document.getElementById("p2-state-select");
                if (stateSelect && stateSelect.value !== station.state) {
                    stateSelect.value = station.state;
                    handleStateChange(false);
                }
                const distSelect = document.getElementById("p2-district-select");
                if (distSelect && distSelect.value !== station.district) {
                    distSelect.value = station.district;
                    handleDistrictChange(false);
                }
                const stSelect = document.getElementById("p2-station-select");
                if (stSelect) {
                    stSelect.value = station.id;
                }
                selectStation(station, true);
            });

            clusterGroup.addLayer(marker);
            stationMarkerMap.set(station.id, marker);
        });
    }

    // ---------------------------------------------------------------------------
    // Filter Handlers
    // ---------------------------------------------------------------------------
    function populateStateFilter() {
        const stateSelect = document.getElementById("p2-state-select");
        if (!stateSelect) return;

        const states = [...new Set(allStations.map(s => s.state))].sort();
        stateSelect.innerHTML = `<option value="">All States (${states.length})</option>`;
        states.forEach(state => {
            const opt = document.createElement("option");
            opt.value = state;
            opt.textContent = state;
            stateSelect.appendChild(opt);
        });

        stateSelect.addEventListener("change", () => handleStateChange(true));
    }

    function handleStateChange(fitBounds = true) {
        const stateSelect = document.getElementById("p2-state-select");
        const districtSelect = document.getElementById("p2-district-select");
        const stationSelect = document.getElementById("p2-station-select");
        const selectedState = stateSelect.value;

        districtSelect.innerHTML = '<option value="">All Districts</option>';
        stationSelect.innerHTML = '<option value="">Select Station</option>';

        let filtered = allStations;
        if (selectedState) {
            filtered = allStations.filter(s => s.state === selectedState);
            const districts = [...new Set(filtered.map(s => s.district))].sort();
            districts.forEach(d => {
                const opt = document.createElement("option");
                opt.value = d;
                opt.textContent = d;
                districtSelect.appendChild(opt);
            });
        }

        renderMapMarkers(filtered);

        if (filtered.length > 0 && phase2Map && fitBounds) {
            const bounds = L.latLngBounds(filtered.map(s => [s.lat, s.lng]));
            phase2Map.fitBounds(bounds, { padding: [40, 40], maxZoom: 10 });
        }
    }

    function handleDistrictChange(fitBounds = true) {
        const stateSelect = document.getElementById("p2-state-select");
        const districtSelect = document.getElementById("p2-district-select");
        const stationSelect = document.getElementById("p2-station-select");

        const selectedState = stateSelect.value;
        const selectedDistrict = districtSelect.value;

        stationSelect.innerHTML = '<option value="">Select Station</option>';

        let filtered = allStations;
        if (selectedState) filtered = filtered.filter(s => s.state === selectedState);
        if (selectedDistrict) filtered = filtered.filter(s => s.district === selectedDistrict);

        filtered.forEach(s => {
            const opt = document.createElement("option");
            opt.value = s.id;
            opt.textContent = `${s.village} (${s.block})`;
            stationSelect.appendChild(opt);
        });

        renderMapMarkers(filtered);

        if (filtered.length > 0 && phase2Map && fitBounds) {
            const bounds = L.latLngBounds(filtered.map(s => [s.lat, s.lng]));
            phase2Map.fitBounds(bounds, { padding: [40, 40], maxZoom: 12 });
        }
    }

    function handleStationDropdownSelect() {
        const stationSelect = document.getElementById("p2-station-select");
        const stationId = stationSelect.value;
        if (!stationId) return;

        const station = allStations.find(s => s.id === stationId);
        if (station) {
            selectStation(station, true);
        }
    }

    // ---------------------------------------------------------------------------
    // Select Station
    // ---------------------------------------------------------------------------
    function selectStation(station, zoomToMarker = true) {
        currentSelectedStation = station;

        // Populate Prediction Year Dropdown dynamically
        populatePredictionYears(station);

        // Update Station Metadata display
        updateStationMetadataDisplay(station, null);

        // Reset previous prediction results until user clicks Predict
        const resultsContainer = document.getElementById("p2-station-results");
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div style="padding:14px;background:#f8fafc;border-radius:6px;border:1px dashed #cbd5e1;text-align:center;">
                    <p style="color:#475569;font-size:0.875rem;">Station selected: <strong>${station.village}</strong></p>
                    <p style="color:#64748b;font-size:0.8rem;margin-top:4px;">Select desired Prediction Year above and click <strong>Predict All Models</strong>.</p>
                </div>
            `;
        }

        // Highlight marker on map
        if (phase2Map && zoomToMarker) {
            phase2Map.setView([station.lat, station.lng], 13);
        }

        if (selectedHighlightMarker && phase2Map) {
            phase2Map.removeLayer(selectedHighlightMarker);
        }

        if (phase2Map) {
            selectedHighlightMarker = L.circleMarker([station.lat, station.lng], {
                radius: 12,
                color: "#1e3a8a",
                weight: 3,
                fillColor: "#3b82f6",
                fillOpacity: 0.4
            }).addTo(phase2Map);
        }
    }

    function populatePredictionYears(station) {
        const yearSelect = document.getElementById("p2-pred-year-select");
        if (!yearSelect) return;

        yearSelect.innerHTML = "";
        const years = station.available_years && station.available_years.length > 0
            ? [...station.available_years]
            : [station.pred || 2025];

        // Sort descending so the most recent target year appears first
        years.sort((a, b) => b - a);

        years.forEach(yr => {
            const opt = document.createElement("option");
            opt.value = yr;
            opt.textContent = `${yr} (uses ${yr - 5}–${yr - 1})`;
            yearSelect.appendChild(opt);
        });

        // Set default to latest year
        yearSelect.value = years[0];
    }

    function updateStationMetadataDisplay(station, seqMeta) {
        const el = document.getElementById("p2-station-metadata");
        if (!el) return;

        const yearSelect = document.getElementById("p2-pred-year-select");
        const predYear = seqMeta ? seqMeta.prediction_year : (yearSelect ? yearSelect.value : (station.pred || 2025));
        const seqRange = seqMeta 
            ? `${seqMeta.sequence_start_year} – ${seqMeta.sequence_end_year}` 
            : `${predYear - 5} – ${predYear - 1}`;

        el.innerHTML = `
            <div class="metadata-grid">
                <div><strong>Station:</strong> ${station.village}</div>
                <div><strong>Block:</strong> ${station.block}</div>
                <div><strong>District:</strong> ${station.district}</div>
                <div><strong>State:</strong> ${station.state}</div>
                <div><strong>Coordinates:</strong> ${station.lat.toFixed(5)}°N, ${station.lng.toFixed(5)}°E</div>
                <div><strong>Historical Sequence:</strong> <span style="font-weight:700;color:#1e40af;">${seqRange}</span></div>
                <div style="grid-column: span 2;"><strong>Target Prediction Year:</strong> <span style="font-weight:800;color:#2563eb;font-size:1rem;">${predYear}</span></div>
            </div>
            <div style="margin-top:8px;font-size:0.75rem;color:#64748b;word-break:break-all;"><strong>ID:</strong> ${station.id}</div>
        `;
    }

    // ---------------------------------------------------------------------------
    // SECTION 1: Predict Existing Monitoring Station (All 3 Models in Parallel)
    // ---------------------------------------------------------------------------
    async function executeStationPrediction() {
        const statusEl = document.getElementById("p2-station-status");
        const resultsContainer = document.getElementById("p2-station-results");
        const historyContainer = document.getElementById("p2-station-history-table");
        const yearSelect = document.getElementById("p2-pred-year-select");

        if (!currentSelectedStation) {
            if (statusEl) statusEl.innerHTML = `<div class="alert alert-error">Please select a monitoring station first.</div>`;
            return;
        }

        const selectedYear = parseInt(yearSelect.value);
        if (isNaN(selectedYear)) {
            if (statusEl) statusEl.innerHTML = `<div class="alert alert-error">Please select a valid prediction year.</div>`;
            return;
        }

        if (statusEl) {
            statusEl.innerHTML = `
                <div class="alert alert-info" style="display:flex;align-items:center;gap:10px;">
                    <div class="loading-spinner"></div>
                    <span>Retrieving historical sequence for Year ${selectedYear} & running LSTM, GRU, and Hybrid models in parallel...</span>
                </div>
            `;
        }

        const encodedId = encodeURIComponent(currentSelectedStation.id);
        const models = ["lstm", "gru", "hybrid"];

        try {
            const promises = models.map(m =>
                fetch(`${API_BASE}/api/phase2/locations/${encodedId}/predict?model=${m}&prediction_year=${selectedYear}`)
                    .then(async res => {
                        if (!res.ok) {
                            const err = await res.json().catch(() => ({}));
                            throw new Error(err.detail || `Prediction failed for ${m.toUpperCase()}`);
                        }
                        return res.json();
                    })
            );

            const [lstmRes, gruRes, hybridRes] = await Promise.all(promises);

            if (statusEl) statusEl.innerHTML = "";

            // Update Metadata card with verified sequence range
            updateStationMetadataDisplay(currentSelectedStation, lstmRes);

            // Render Results Table
            renderResultsTable(resultsContainer, lstmRes, gruRes, hybridRes, selectedYear);

            // Render Historical Sequence Table in collapsible section
            if (lstmRes.historical_sequence && historyContainer) {
                renderHistoricalTable(historyContainer, lstmRes.historical_sequence, selectedYear);
            }

        } catch (err) {
            console.error("Existing Station prediction failed:", err);
            if (statusEl) {
                statusEl.innerHTML = `<div class="alert alert-error">${err.message}</div>`;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Render Results Table (Consistent for both sections)
    // ---------------------------------------------------------------------------
    function renderResultsTable(container, lstmRes, gruRes, hybridRes, predYear) {
        if (!container) return;

        const lstmRisk = getRiskCategory(lstmRes.predicted_wl);
        const gruRisk = getRiskCategory(gruRes.predicted_wl);
        const hybridRisk = getRiskCategory(hybridRes.predicted_wl);

        container.innerHTML = `
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Model Architecture</th>
                        <th>Predicted Depth (mbgl)</th>
                        <th>Risk Category</th>
                        <th>Input Modality</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>LSTM</strong></td>
                        <td><span style="font-size:1.15rem;font-weight:800;color:#1e3a8a;">${lstmRes.predicted_wl.toFixed(2)}</span> mbgl</td>
                        <td><span class="risk-badge ${lstmRisk.className}">${lstmRisk.label}</span></td>
                        <td style="font-size:0.8rem;color:#64748b;">5-Year Temporal (4 features)</td>
                    </tr>
                    <tr>
                        <td><strong>GRU</strong></td>
                        <td><span style="font-size:1.15rem;font-weight:800;color:#1e3a8a;">${gruRes.predicted_wl.toFixed(2)}</span> mbgl</td>
                        <td><span class="risk-badge ${gruRisk.className}">${gruRisk.label}</span></td>
                        <td style="font-size:0.8rem;color:#64748b;">5-Year Temporal (4 features)</td>
                    </tr>
                    <tr>
                        <td><strong>Hybrid Spatio-Temporal GRU</strong></td>
                        <td><span style="font-size:1.15rem;font-weight:800;color:#1e3a8a;">${hybridRes.predicted_wl.toFixed(2)}</span> mbgl</td>
                        <td><span class="risk-badge ${hybridRisk.className}">${hybridRisk.label}</span></td>
                        <td style="font-size:0.8rem;color:#64748b;">Temporal + Coordinates (${hybridRes.latitude.toFixed(4)}°, ${hybridRes.longitude.toFixed(4)}°)</td>
                    </tr>
                </tbody>
            </table>
            <div style="margin-top:10px;font-size:0.75rem;color:#64748b;text-align:right;">
                Target Forecast Year: <strong>${predYear}</strong> | Input Sequence: <strong>${lstmRes.sequence_start_year}–${lstmRes.sequence_end_year}</strong>
            </div>
        `;
    }

    // ---------------------------------------------------------------------------
    // Render Historical Sequence Table
    // ---------------------------------------------------------------------------
    function renderHistoricalTable(container, sequence, predYear) {
        let rowsHtml = sequence.map((row, i) => `
            <tr>
                <td style="font-weight:700;">Year ${i + 1} (${row.year})</td>
                <td><strong>${row.wl.toFixed(2)}</strong></td>
                <td>${row.rainfall.toFixed(1)}</td>
                <td>${row.temperature.toFixed(1)}</td>
                <td>${row.humidity.toFixed(1)}</td>
            </tr>
        `).join("");

        container.innerHTML = `
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Observation Year</th>
                            <th>Groundwater Depth (mbgl)</th>
                            <th>Rainfall (mm)</th>
                            <th>Temperature (°C)</th>
                            <th>Humidity (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rowsHtml}
                        <tr style="background:#f0fdf4;border-top:2px solid #86efac;font-weight:bold;">
                            <td style="color:#166534;">Target Forecast (${predYear})</td>
                            <td colspan="4" style="color:#166534;">
                                Predicted by Deep Learning Models (LSTM / GRU / Hybrid)
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        `;
    }

    // ---------------------------------------------------------------------------
    // SECTION 2: New Location / New Well Prediction (All 3 Models in Parallel)
    // ---------------------------------------------------------------------------
    async function executeNewLocationPrediction() {
        const statusEl = document.getElementById("p2-new-status");
        const resultsContainer = document.getElementById("p2-new-results");

        const lat = parseFloat(document.getElementById("p2_new_lat").value);
        const lon = parseFloat(document.getElementById("p2_new_lon").value);

        if (isNaN(lat) || isNaN(lon)) {
            statusEl.innerHTML = `<div class="alert alert-error">Please enter valid numeric Latitude and Longitude.</div>`;
            return;
        }

        // Collect and validate 5 years of observations
        const years = [];
        const temporalSequence = [];

        for (let i = 0; i < 5; i++) {
            const yr = parseInt(document.getElementById(`p2_new_yr_${i}`).value);
            const wl = parseFloat(document.getElementById(`p2_new_wl_${i}`).value);
            const rain = parseFloat(document.getElementById(`p2_new_rain_${i}`).value);
            const temp = parseFloat(document.getElementById(`p2_new_temp_${i}`).value);
            const hum = parseFloat(document.getElementById(`p2_new_hum_${i}`).value);

            if (isNaN(yr) || isNaN(wl) || isNaN(rain) || isNaN(temp) || isNaN(hum)) {
                statusEl.innerHTML = `<div class="alert alert-error">Please fill in all five observation rows with valid numbers.</div>`;
                return;
            }

            years.push(yr);
            temporalSequence.push([wl, rain, temp, hum]);
        }

        // Validate that years are consecutive
        for (let i = 0; i < 4; i++) {
            if (years[i + 1] !== years[i] + 1) {
                statusEl.innerHTML = `<div class="alert alert-error">Historical observation years must be consecutive (e.g. ${years[0]}, ${years[0] + 1}, ${years[0] + 2}, ${years[0] + 3}, ${years[0] + 4}). Found gap between ${years[i]} and ${years[i + 1]}.</div>`;
                return;
            }
        }

        // Validate target prediction year is Year 5 + 1
        const predYear = parseInt(document.getElementById("p2_new_pred_year").value);
        const expectedPredYear = years[4] + 1;
        if (isNaN(predYear) || predYear !== expectedPredYear) {
            statusEl.innerHTML = `<div class="alert alert-error">Target Prediction Year must be exactly one year after the fifth observation year (${expectedPredYear}). Got ${predYear}.</div>`;
            return;
        }

        statusEl.innerHTML = `
            <div class="alert alert-info" style="display:flex;align-items:center;gap:10px;">
                <div class="loading-spinner"></div>
                <span>Executing LSTM, GRU, and Hybrid models in parallel for Target Year ${predYear}...</span>
            </div>
        `;

        try {
            // Run all 3 models in parallel via POST /api/phase2/predict
            const pLstm = fetch(`${API_BASE}/api/phase2/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: "lstm", temporal_sequence: temporalSequence })
            }).then(r => r.json());

            const pGru = fetch(`${API_BASE}/api/phase2/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: "gru", temporal_sequence: temporalSequence })
            }).then(r => r.json());

            const pHybrid = fetch(`${API_BASE}/api/phase2/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: "hybrid", temporal_sequence: temporalSequence, latitude: lat, longitude: lon })
            }).then(r => r.json());

            const [lstmRes, gruRes, hybridRes] = await Promise.all([pLstm, pGru, pHybrid]);

            statusEl.innerHTML = "";

            // Format responses to match renderResultsTable structure
            const lstmObj = {
                predicted_wl: lstmRes.predicted_wl,
                sequence_start_year: years[0],
                sequence_end_year: years[4]
            };
            const gruObj = {
                predicted_wl: gruRes.predicted_wl,
                sequence_start_year: years[0],
                sequence_end_year: years[4]
            };
            const hybridObj = {
                predicted_wl: hybridRes.predicted_wl,
                latitude: lat,
                longitude: lon,
                sequence_start_year: years[0],
                sequence_end_year: years[4]
            };

            renderResultsTable(resultsContainer, lstmObj, gruObj, hybridObj, predYear);

        } catch (err) {
            console.error("New Location prediction error:", err);
            statusEl.innerHTML = `<div class="alert alert-error">Prediction failed: ${err.message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // Preset Loader: Load current station data into New Location section
    // ---------------------------------------------------------------------------
    async function loadPresetIntoNewLocation() {
        const statusEl = document.getElementById("p2-new-status");
        if (!currentSelectedStation) {
            statusEl.innerHTML = `<div class="alert alert-error">Please select a monitoring station first.</div>`;
            return;
        }

        const encodedId = encodeURIComponent(currentSelectedStation.id);
        const yearSelect = document.getElementById("p2-pred-year-select");
        const predYear = yearSelect ? yearSelect.value : (currentSelectedStation.pred || 2025);

        try {
            const res = await fetch(`${API_BASE}/api/phase2/locations/${encodedId}/predict?model=gru&prediction_year=${predYear}`);
            if (!res.ok) throw new Error("Could not retrieve station data for preset.");
            const data = await res.json();

            // Populate Coordinates
            document.getElementById("p2_new_lat").value = data.latitude.toFixed(4);
            document.getElementById("p2_new_lon").value = data.longitude.toFixed(4);

            // Populate 5 observation rows
            if (data.historical_sequence && data.historical_sequence.length === 5) {
                data.historical_sequence.forEach((row, i) => {
                    const yrInput = document.getElementById(`p2_new_yr_${i}`);
                    const wlInput = document.getElementById(`p2_new_wl_${i}`);
                    const rainInput = document.getElementById(`p2_new_rain_${i}`);
                    const tempInput = document.getElementById(`p2_new_temp_${i}`);
                    const humInput = document.getElementById(`p2_new_hum_${i}`);

                    if (yrInput) yrInput.value = row.year;
                    if (wlInput) wlInput.value = row.wl.toFixed(2);
                    if (rainInput) rainInput.value = row.rainfall.toFixed(1);
                    if (tempInput) tempInput.value = row.temperature.toFixed(1);
                    if (humInput) humInput.value = row.humidity.toFixed(1);
                });
            }

            // Set Target Prediction Year
            document.getElementById("p2_new_pred_year").value = data.prediction_year;

            statusEl.innerHTML = `<div class="alert alert-info">Loaded ${currentSelectedStation.village} (${data.sequence_start_year}–${data.sequence_end_year}) sequence into form.</div>`;
            setTimeout(() => { if (statusEl.innerHTML.includes("Loaded")) statusEl.innerHTML = ""; }, 4000);

        } catch (err) {
            statusEl.innerHTML = `<div class="alert alert-error">${err.message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // Quick-Action Test Station Button
    // ---------------------------------------------------------------------------
    function initQuickButtons() {
        const testBtn = document.getElementById("p2-quick-test-btn");
        if (testBtn) {
            testBtn.addEventListener("click", () => {
                const testStation = allStations.find(s => s.id === TEST_STATION_ID);
                if (testStation) {
                    const stateSelect = document.getElementById("p2-state-select");
                    if (stateSelect) {
                        stateSelect.value = testStation.state;
                        handleStateChange(false);
                    }
                    const districtSelect = document.getElementById("p2-district-select");
                    if (districtSelect) {
                        districtSelect.value = testStation.district;
                        handleDistrictChange(false);
                    }
                    const stationSelect = document.getElementById("p2-station-select");
                    if (stationSelect) {
                        stationSelect.value = testStation.id;
                    }
                    selectStation(testStation, true);
                }
            });
        }

        const distSelect = document.getElementById("p2-district-select");
        if (distSelect) {
            distSelect.addEventListener("change", () => handleDistrictChange(true));
        }

        const stSelect = document.getElementById("p2-station-select");
        if (stSelect) {
            stSelect.addEventListener("change", handleStationDropdownSelect);
        }

        // Section 1 Predict Button
        const stationPredictBtn = document.getElementById("p2-station-predict-btn");
        if (stationPredictBtn) {
            stationPredictBtn.addEventListener("click", executeStationPrediction);
        }

        // Section 2 Predict Button
        const newPredictBtn = document.getElementById("p2-new-predict-btn");
        if (newPredictBtn) {
            newPredictBtn.addEventListener("click", executeNewLocationPrediction);
        }

        // Section 2 Preset Load Button
        const presetBtn = document.getElementById("p2-new-load-preset-btn");
        if (presetBtn) {
            presetBtn.addEventListener("click", loadPresetIntoNewLocation);
        }

        // Update sequence preview on Year change
        const yearSelect = document.getElementById("p2-pred-year-select");
        if (yearSelect) {
            yearSelect.addEventListener("change", () => {
                if (currentSelectedStation) {
                    updateStationMetadataDisplay(currentSelectedStation, null);
                }
            });
        }
    }

    // ---------------------------------------------------------------------------
    // DOM Ready Initialization
    // ---------------------------------------------------------------------------
    document.addEventListener("DOMContentLoaded", () => {
        initMap();
        loadStations();
        initQuickButtons();
    });

})();
