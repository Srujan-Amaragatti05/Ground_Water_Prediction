// const API_URL = "http://localhost:8000/predict";
let manualMarker = null;

/* Utility: show error */
function showManualError(msg) {
    document.getElementById("manualResult").innerHTML =
        `<span style="color:red;">${msg}</span>`;
}

/* Handle Predict button */
document.getElementById("manualPredictBtn").addEventListener("click", async () => {

    const district = document.getElementById("m_district").value.trim();
    const block = document.getElementById("m_block").value.trim();
    const latitude = parseFloat(document.getElementById("m_latitude").value);
    const longitude = parseFloat(document.getElementById("m_longitude").value);
    const year = parseInt(document.getElementById("m_year").value);
    const month = parseInt(document.getElementById("m_month").value);

    // console.log("input done");
    /* Basic validation */
    if (!district || !block || isNaN(latitude) || isNaN(longitude) || isNaN(year) || isNaN(month)) {
        showManualError("Please fill all fields correctly.");
        return;
    }
    // console.log("ckeck done");

    document.getElementById("manualResult").innerHTML = "⏳ Predicting...";

    const payload = {
        Year: year,
        Month: month,
        LATITUDE: latitude,
        LONGITUDE: longitude,
        DISTRICT: district,
        BLOCK: block
    };

    // console.log("send done");

    function getColor(wl) {
        if (wl <= 5) return "blue";
        if (wl <= 10) return "yellow";
        return "red";
    }
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Prediction failed");

        const result = await response.json();

        /* Show prediction result */
        document.getElementById("manualResult").innerHTML = `
            Predicted WL: ${result.predicted_WL.toFixed(2)} mbgl <br>
            Risk Category: ${result.risk_category}
        `;

        /* Remove previous manual marker */
        if (manualMarker) {
            map.removeLayer(manualMarker);
        }

        /* Add highlighted marker */
        manualMarker = L.circleMarker(
            [latitude, longitude],
            {
                radius: 10,
                color: getColor(result.predicted_WL),
                fillColor: getColor(result.predicted_WL),
                fillOpacity: 0.9,
                weight: 2
            }
        )
            .bindPopup(`
            <b>Manual Prediction</b><br>
            WL: ${result.predicted_WL.toFixed(2)} mbgl<br>
            Risk: ${result.risk_category}
        `)
            .addTo(map);

        map.setView([latitude, longitude], 11);

    } catch (err) {
        showManualError("Unable to fetch prediction.");
        console.error(err);
    }
});
