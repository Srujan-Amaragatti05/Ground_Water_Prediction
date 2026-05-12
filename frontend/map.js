const API_URL = "http://localhost:8000/predict";
const detailsDiv = document.getElementById("details");

async function fetchPrediction(input) {
    const response = await fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(input)
    });

    if (!response.ok) {
        throw new Error("Prediction API failed");
    }

    return await response.json();
}

/* -------------------------------
   Map initialization
-------------------------------- */
const map = L.map("map", { preferCanvas: true })
    .setView([15.5, 80.5], 7);

/* Base map */
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors"
}).addTo(map);

/* Color based on groundwater depth */
function getColor(wl) {
    if (wl <= 5) return "blue";
    if (wl <= 10) return "yellow";
    return "red";
}

/* Cluster group */
const clusterGroup = L.markerClusterGroup({
    disableClusteringAtZoom: 13,
    showCoverageOnHover: false
});
map.addLayer(clusterGroup);

const yearSelect = document.getElementById("yearSelect");
let allData = [];

/* -------------------------------
   Load groundwater data
-------------------------------- */
fetch("wells_data_years.json")
    .then(res => res.json())
    .then(data => {
        allData = data;

        /* Extract unique years */
        const years = [...new Set(data.map(d => d.year))].sort();

        /* Populate dropdown */
        years.forEach(year => {
            const opt = document.createElement("option");
            opt.value = year;
            opt.textContent = year;
            yearSelect.appendChild(opt);
        });

        /* Default = latest year */
        const latestYear = years[years.length - 1];
        yearSelect.value = latestYear;

        plotYear(latestYear);
    });

/* -------------------------------
   Plot wells for selected year
-------------------------------- */
function plotYear(selectedYear) {
    clusterGroup.clearLayers();

    const bounds = [];
    const filtered = allData.filter(d => d.year == selectedYear);

    filtered.forEach(well => {
        bounds.push([well.latitude, well.longitude]);

        L.circleMarker(
            [well.latitude, well.longitude],
            {
                radius: 6,
                weight: 1,
                color: getColor(well.wl),
                fillColor: getColor(well.wl),
                fillOpacity: 0.7
            }
        )
            .on("click", async () => {

                const selectedYear = Number(yearSelect.value);

                // Show loading
                detailsDiv.innerHTML = `
        <b>Well Details</b><br>
        District: ${well.district}<br>
        Block: ${well.block}<br>
        Village: ${well.village}<br>
        <br>
        ⏳ Fetching predictions...
    `;

                try {
                    console.log("Fetching predictions...");
                    // Predict for +1 year
                    const pred1 = await fetchPrediction({
                        Year: selectedYear + 1,
                        Month: 6,
                        LATITUDE: Number(well.latitude),
                        LONGITUDE: Number(well.longitude),
                        DISTRICT: String(well.district),
                        BLOCK: String(well.block)
                    });


                    // Predict for +2 year
                    const pred2 = await fetchPrediction({
                        Year: selectedYear + 2,
                        Month: 6,
                        LATITUDE: Number(well.latitude),
                        LONGITUDE: Number(well.longitude),
                        DISTRICT: String(well.district),
                        BLOCK: String(well.block)
                    });


                    // Update details panel
                    detailsDiv.innerHTML = `
            <b>Well Details</b><br><br>

            <b>Location</b><br>
            District: ${well.district}<br>
            Block: ${well.block}<br>
            Village: ${well.village}<br><br>

            <b>Current WL (${selectedYear})</b><br>
            ${well.wl} mbgl<br><br>

            <b>Predicted WL</b><br>
            ${selectedYear + 1}: ${pred1.predicted_WL.toFixed(2)} mbgl<br>
            ${selectedYear + 2}: ${pred2.predicted_WL.toFixed(2)} mbgl<br><br>

            <b>Risk Category</b><br>
            ${pred2.risk_category}
        `;
                } catch (err) {
                    detailsDiv.innerHTML = `
            <b>Error</b><br>
            Unable to fetch prediction.<br>
            Please check backend service.
        `;
                    console.error(err);
                }
            })
            .addTo(clusterGroup);

    });

    if (bounds.length) {
        map.fitBounds(bounds);
        map.invalidateSize();
    }
}

/* -------------------------------
   Year change handler
-------------------------------- */
yearSelect.addEventListener("change", e => {
    plotYear(e.target.value);
});

/* -------------------------------
   Legend
-------------------------------- */
const legend = L.control({ position: "bottomright" });

legend.onAdd = function () {
    const div = L.DomUtil.create("div", "legend");
    div.innerHTML = `
        <b>Groundwater Level</b><br>
        <span style="background: blue"></span> Shallow (≤ 5 mbgl)<br>
        <span style="background: yellow"></span> Moderate (5–10 mbgl)<br>
        <span style="background: red"></span> Deep (> 10 mbgl)
    `;
    return div;
};

legend.addTo(map);
