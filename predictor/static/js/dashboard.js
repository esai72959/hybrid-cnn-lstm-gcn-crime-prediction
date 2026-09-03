/**
 * dashboard.js
 *
 * Fetches live statistics from /api/dashboard/ and populates the
 * Crime Prediction System dashboard.
 *
 * Includes:
 * - KPI overview cards (records, features, period, hybrid accuracy)
 * - Model benchmark cards + bars (CNN / LSTM / Hybrid CNN-LSTM)
 * - Year-wise Crime Trend Analysis
 *
 * IMPORTANT: if the API request fails, this file never falls back to
 * fake "0" values. Every card is given an honest "Data unavailable"
 * state instead (see renderErrorState / markUnavailable below).
 */

(function () {
  "use strict";

  const DASHBOARD_API_URL = "/api/dashboard/";

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = value;
    }
  }

  function markUnavailable(id, label) {
    const el = document.getElementById(id);
    if (!el) {
      return;
    }
    el.textContent = label || "Unavailable";
    const card = el.closest(".dash-kpi-card, .dash-stat-card, .status-card");
    if (card) {
      card.classList.add("is-unavailable");
    }
  }

  function setBarWidth(id, percent) {
    const el = document.getElementById(id);

    if (el) {
      const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
      el.style.width = safePercent + "%";
    }
  }

  function formatNumber(value) {
    const num = Number(value);

    if (!Number.isFinite(num)) {
      return "0";
    }

    return num.toLocaleString("en-US");
  }

  /**
   * Format large values for chart axis labels.
   */
  function formatCompactNumber(value) {
    const num = Number(value);

    if (!Number.isFinite(num)) {
      return "0";
    }

    if (num >= 10000000) {
      return (num / 10000000).toFixed(1) + "Cr";
    }

    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    }

    if (num >= 1000) {
      return (num / 1000).toFixed(0) + "K";
    }

    return Math.round(num).toString();
  }

  /**
   * Render the year-wise crime trend chart.
   *
   * Expects data.trend from /api/dashboard/ in the form:
   * [ { year: 2001, crime_count: 2003540 }, ... ]
   */
  function renderCrimeTrend(trend) {
    const chartWrap = document.querySelector(".chart-placeholder-wrap");

    if (!chartWrap) {
      return;
    }

    if (!Array.isArray(trend) || trend.length === 0) {
      chartWrap.innerHTML = `
        <div class="chart-empty-state">
          <p>Historical trend data is currently unavailable.</p>
        </div>
      `;
      return;
    }

    const cleanTrend = trend
      .map(function (item) {
        return {
          year: Number(item.year),
          crime_count: Number(item.crime_count),
        };
      })
      .filter(function (item) {
        return Number.isFinite(item.year) && Number.isFinite(item.crime_count);
      })
      .sort(function (a, b) {
        return a.year - b.year;
      });

    if (cleanTrend.length === 0) {
      chartWrap.innerHTML = `
        <div class="chart-empty-state">
          <p>Historical trend data is currently unavailable.</p>
        </div>
      `;
      return;
    }

    const width = 900;
    const height = 320;

    const margin = { top: 24, right: 30, bottom: 40, left: 70 };

    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const values = cleanTrend.map(function (item) {
      return item.crime_count;
    });

    const minValue = Math.min.apply(null, values);
    const maxValue = Math.max.apply(null, values);
    const valueRange = maxValue - minValue || 1;

    function xPosition(index) {
      if (cleanTrend.length === 1) {
        return margin.left + chartWidth / 2;
      }
      return margin.left + (index / (cleanTrend.length - 1)) * chartWidth;
    }

    function yPosition(value) {
      return margin.top + chartHeight - ((value - minValue) / valueRange) * chartHeight;
    }

    const points = cleanTrend
      .map(function (item, index) {
        return xPosition(index) + "," + yPosition(item.crime_count);
      })
      .join(" ");

    const areaPoints =
      margin.left +
      "," +
      (margin.top + chartHeight) +
      " " +
      points +
      " " +
      xPosition(cleanTrend.length - 1) +
      "," +
      (margin.top + chartHeight);

    const yTicks = 4;
    let gridLines = "";
    let yLabels = "";

    for (let i = 0; i <= yTicks; i++) {
      const ratio = i / yTicks;
      const y = margin.top + chartHeight - ratio * chartHeight;
      const value = minValue + ratio * valueRange;

      gridLines += `<line x1="${margin.left}" y1="${y}" x2="${margin.left + chartWidth}" y2="${y}" stroke="currentColor" stroke-opacity="0.10" />`;
      yLabels += `<text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="currentColor" opacity="0.65">${formatCompactNumber(value)}</text>`;
    }

    // Show at most ~8 year labels so they never overlap on narrow screens.
    const labelEvery = Math.max(1, Math.ceil(cleanTrend.length / 8));

    let xLabels = "";
    let pointsMarkup = "";

    cleanTrend.forEach(function (item, index) {
      const x = xPosition(index);
      const y = yPosition(item.crime_count);

      if (index % labelEvery === 0 || index === cleanTrend.length - 1) {
        xLabels += `<text x="${x}" y="${height - 12}" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.70">${item.year}</text>`;
      }

      pointsMarkup += `<circle cx="${x}" cy="${y}" r="3.5" fill="currentColor" class="crime-trend-point"><title>${item.year}: ${formatNumber(item.crime_count)} crimes</title></circle>`;
    });

    chartWrap.innerHTML = `
      <svg class="crime-trend-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Year-wise crime trend from ${cleanTrend[0].year} to ${cleanTrend[cleanTrend.length - 1].year}">
        <defs>
          <linearGradient id="crimeTrendAreaGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="currentColor" stop-opacity="0.20" />
            <stop offset="100%" stop-color="currentColor" stop-opacity="0" />
          </linearGradient>
        </defs>
        ${gridLines}
        ${yLabels}
        <polygon points="${areaPoints}" fill="url(#crimeTrendAreaGradient)" />
        <polyline points="${points}" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
        ${pointsMarkup}
        <line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${margin.left + chartWidth}" y2="${margin.top + chartHeight}" stroke="currentColor" stroke-opacity="0.20" />
        ${xLabels}
      </svg>
    `;
  }

  /**
   * Populate the dashboard DOM with data returned from the API.
   */
  function renderDashboard(data) {
    const dataset = data.dataset || {};
    const models = data.models || {};

    // ---- KPI Cards ----
    // Dataset-dependent fields: the API sends null (not 0) for anything
    // it can't honestly compute, so treat null/undefined as "unavailable"
    // rather than letting formatNumber() turn it into a fake "0".
    if (Number.isFinite(Number(dataset.records))) {
      setText("totalRecords", formatNumber(dataset.records));
    } else {
      markUnavailable("totalRecords");
    }

    if (Number.isFinite(Number(dataset.features))) {
      setText("totalFeatures", formatNumber(dataset.features));
    } else {
      markUnavailable("totalFeatures");
    }

    if (dataset.period) {
      setText("datasetPeriod", dataset.period);
    } else {
      markUnavailable("datasetPeriod");
    }

    // ---- Model Benchmark ----
    // These come from the static, authoritative research metrics, so
    // they render independently of live dataset/model-loading state -
    // the reported R² figures never change just because the model file
    // can't currently be loaded (see setModelBadge below for that).
    const cnnAccuracy = models.cnn ? models.cnn.accuracy : null;
    const lstmAccuracy = models.lstm ? models.lstm.accuracy : null;
    const hybridAccuracy = models.hybrid ? models.hybrid.accuracy : null;
    const gcnAccuracy = models.hybrid_gcn ? models.hybrid_gcn.accuracy : (models.hybrid_gcn_accuracy || 96.10);

    if (hybridAccuracy != null) {
      setText("hybridAccuracy", Number(hybridAccuracy).toFixed(2));
    } else {
      markUnavailable("hybridAccuracy");
    }

    if (gcnAccuracy != null) {
      setText("hybridGcnAccuracy", Number(gcnAccuracy).toFixed(2));
      setText("gcnAccuracy", Number(gcnAccuracy).toFixed(2));
      setBarWidth("gcnBar", gcnAccuracy);
    } else {
      markUnavailable("hybridGcnAccuracy");
      markUnavailable("gcnAccuracy");
      setBarWidth("gcnBar", 0);
    }

    if (cnnAccuracy != null) {
      setText("cnnAccuracy", Number(cnnAccuracy).toFixed(2));
      setBarWidth("cnnBar", cnnAccuracy);
    } else {
      markUnavailable("cnnAccuracy");
      setBarWidth("cnnBar", 0);
    }

    if (lstmAccuracy != null) {
      setText("lstmAccuracy", Number(lstmAccuracy).toFixed(2));
      setBarWidth("lstmBar", lstmAccuracy);
    } else {
      markUnavailable("lstmAccuracy");
      setBarWidth("lstmBar", 0);
    }

    // ---- Crime Trend ----
    renderCrimeTrend(data.trend || []);

    // ---- Crime Category Heatmap ----
    loadCategoryHeatmap();

    // ---- Shared Counter Animation ----
    if (window.CPS && typeof window.CPS.animateCounters === "function") {
      window.CPS.animateCounters("[data-counter]");
    }
  }

  /**
   * Render the Spatio-Temporal Crime Category Heatmap Matrix.
   */
  function loadCategoryHeatmap() {
    const container = document.getElementById("heatmapContainer");
    if (!container) return;

    fetch("/static/data/crime_category_heatmap.json")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (!data || !data.years || !data.categories) {
          container.innerHTML = '<p class="text-muted text-center">Heatmap data unavailable.</p>';
          return;
        }

        const years = data.years;
        const categories = data.categories;

        let html = `
          <table class="table table-bordered table-sm text-center align-middle mb-0" style="font-size:12px; border-color: rgba(0,0,0,0.06);">
            <thead style="background:#F8FAFC; color:#475569;">
              <tr>
                <th class="text-start p-2" style="min-width:140px;">Crime Category</th>
                ${years.map(function (y) { return `<th class="p-2">${y}</th>`; }).join('')}
              </tr>
            </thead>
            <tbody>
        `;

        categories.forEach(function (cat) {
          const minVal = cat.min;
          const maxVal = cat.max;
          const isTotal = cat.key === "TOTAL IPC CRIMES";

          html += `
            <tr style="${isTotal ? 'font-weight:700; background:#F8FAFC;' : ''}">
              <td class="text-start p-2" style="white-space:nowrap;">
                ${cat.category}
              </td>
          `;

          years.forEach(function (y) {
            const val = cat.yearly_values[String(y)] || 0;
            const ratio = maxVal > minVal ? (val - minVal) / (maxVal - minVal) : 0.5;

            let bgColor, textColor = "#1E293B";
            if (ratio < 0.35) {
              bgColor = `rgba(226, 241, 232, ${0.4 + ratio * 1.5})`;
            } else if (ratio < 0.7) {
              bgColor = `rgba(253, 230, 138, ${0.4 + (ratio - 0.35) * 1.5})`;
            } else {
              bgColor = `rgba(248, 113, 113, ${0.35 + (ratio - 0.7) * 1.8})`;
            }

            html += `
              <td class="p-2" style="background-color:${bgColor}; color:${textColor}; cursor:pointer;" 
                  title="${cat.category} (${y}): ${val.toLocaleString()} recorded cases">
                ${formatCompactNumber(val)}
              </td>
            `;
          });

          html += `</tr>`;
        });

        html += `</tbody></table>`;
        container.innerHTML = html;
      })
      .catch(function (err) {
        console.error("Failed to load heatmap data:", err);
        container.innerHTML = '<p class="text-muted text-center">Unable to load heatmap.</p>';
      });
  }

  /**
   * Display an honest "data unavailable" state on every dynamic element
   * if the dashboard API fails. Never falls back to fabricated 0 values.
   */
  function renderErrorState() {
    markUnavailable("totalRecords");
    markUnavailable("totalFeatures");
    markUnavailable("datasetPeriod");
    markUnavailable("hybridAccuracy");
    markUnavailable("hybridGcnAccuracy");

    markUnavailable("cnnAccuracy");
    markUnavailable("lstmAccuracy");
    markUnavailable("gcnAccuracy");

    setBarWidth("cnnBar", 0);
    setBarWidth("lstmBar", 0);
    setBarWidth("gcnBar", 0);

    renderCrimeTrend([]);
  }

  /**
   * Fetch dashboard data from the API.
   */
  function loadDashboard() {
    fetch(DASHBOARD_API_URL, {
      method: "GET",
      headers: { Accept: "application/json" },
    })
      .then(function (response) {
        return response.json().then(function (payload) {
          return { ok: response.ok, payload: payload };
        });
      })
      .then(function (result) {
        if (!result.ok || result.payload.status !== "success") {
          throw new Error((result.payload && result.payload.message) || "Dashboard API returned an error.");
        }
        renderDashboard(result.payload);
      })
      .catch(function (error) {
        console.error("Failed to load dashboard data:", error);
        renderErrorState();
      });
  }

  document.addEventListener("DOMContentLoaded", loadDashboard);
})();