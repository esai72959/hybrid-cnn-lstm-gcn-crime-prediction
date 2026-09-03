/**
 * forecast.js
 * ---------------------------------------------------------------------------
 * Wires up the existing Forecast page markup (IDs unchanged):
 *   #forecastInputForm, #forecastState, #forecastDistrict, #forecastYear,
 *   #forecastResults, #forecastAlert, #generateForecastBtn,
 *   #fcState #fcDistrict #fcYear #fcModel #fcPredictedCount #fcRiskLevel
 *   #fcRiskScore #fcConfidence #fcRecommendation,
 *   #forecastChartEmpty, #forecastChartWrap, #forecastTrendChart
 *
 * Data sources:
 *   GET  /api/states/            -> {"status":"success","states":[...]}
 *   GET  /api/districts/?state=  -> [...]  (existing api_districts view)
 *   POST /api/predict/           -> existing api_predict() response
 *   GET  /api/forecast-trend/?state=&district= -> returns {status, years: [], values: []}
 * ---------------------------------------------------------------------------
 */

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('forecastInputForm');
  const stateSelect = document.getElementById('forecastState');
  const districtSelect = document.getElementById('forecastDistrict');
  const yearSelect = document.getElementById('forecastYear');
  const submitBtn = document.getElementById('generateForecastBtn');
  const alertBox = document.getElementById('forecastAlert');
  const resultsSection = document.getElementById('forecastResults');
  const chartEmptyState = document.getElementById('forecastChartEmpty');
  const chartWrap = document.getElementById('forecastChartWrap');
  const chartCanvas = document.getElementById('forecastTrendChart');

  if (!form || !stateSelect || !districtSelect || !yearSelect) {
    return;
  }

  // -------------------------------------------------------------------
  // CSRF helper (Django's documented cookie-based CSRF pattern)
  // -------------------------------------------------------------------
  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }
  const csrftoken = getCookie('csrftoken');

  // -------------------------------------------------------------------
  // Alert / loading helpers
  // -------------------------------------------------------------------
  function showError(message) {
    if (!alertBox) return;
    alertBox.textContent = message;
    alertBox.classList.remove('d-none');
    alertBox.setAttribute('role', 'alert');
  }

  function clearError() {
    if (!alertBox) return;
    alertBox.textContent = '';
    alertBox.classList.add('d-none');
  }

  function setLoading(isLoading) {
    if (!submitBtn) return;
    submitBtn.disabled = isLoading;
    submitBtn.innerHTML = isLoading
      ? '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Generating Forecast...'
      : '<i class="bi bi-magic me-2"></i>Generate Forecast';
  }

  function showChartEmptyState() {
    if (chartEmptyState) chartEmptyState.classList.remove('d-none');
    if (chartWrap) chartWrap.classList.add('d-none');
  }

  function showChart() {
    if (chartEmptyState) chartEmptyState.classList.add('d-none');
    if (chartWrap) chartWrap.classList.remove('d-none');
  }

  // -------------------------------------------------------------------
  // Forecast Year options
  // -------------------------------------------------------------------
  function populateYears() {
    const years = [2025, 2026, 2027, 2028, 2029, 2030];
    yearSelect.innerHTML = '<option value="">Select Year</option>';
    years.forEach((y) => {
      const opt = document.createElement('option');
      opt.value = y;
      opt.textContent = y;
      yearSelect.appendChild(opt);
    });
  }

  // -------------------------------------------------------------------
  // States loader
  // -------------------------------------------------------------------
  let stateGroups = {};
  let districtStateMap = {};

  function normalizeKey(name) {
    return name.replace(/\s+/g, '').toUpperCase();
  }

  function hasSpacedAmpersand(name) {
    return /\s&\s/.test(name);
  }

  function buildStateGroups(rawStates) {
    const groups = {};
    (rawStates || []).forEach((raw) => {
      const name = String(raw).trim();
      if (!name) return;
      const key = normalizeKey(name);
      if (!groups[key]) {
        groups[key] = { display: name, variants: [name] };
      } else {
        groups[key].variants.push(name);
        if (hasSpacedAmpersand(name) && !hasSpacedAmpersand(groups[key].display)) {
          groups[key].display = name;
        }
      }
    });
    return groups;
  }

  function loadStates() {
    fetch('/api/states/')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load states.');
        return res.json();
      })
      .then((data) => {
        const rawStates = Array.isArray(data) ? data : data.states;
        if (!Array.isArray(rawStates)) {
          throw new Error('Unexpected response format from /api/states/.');
        }

        stateGroups = buildStateGroups(rawStates);

        const keys = Object.keys(stateGroups).sort((a, b) =>
          stateGroups[a].display.localeCompare(stateGroups[b].display)
        );

        stateSelect.innerHTML = '<option value="">Select State</option>';
        keys.forEach((key) => {
          const opt = document.createElement('option');
          opt.value = key;
          opt.textContent = stateGroups[key].display;
          stateSelect.appendChild(opt);
        });
      })
      .catch(() => {
        stateSelect.innerHTML = '<option value="">Unable to load states</option>';
        showError('Could not load the list of states. Please refresh the page and try again.');
      });
  }

  // -------------------------------------------------------------------
  // Districts loader
  // -------------------------------------------------------------------
  function loadDistricts(stateKey) {
    const group = stateGroups[stateKey];
    districtSelect.disabled = true;
    districtSelect.innerHTML = '<option value="">Loading Districts...</option>';
    districtStateMap = {};

    if (!group) {
      districtSelect.innerHTML = '<option value="">Select State First</option>';
      return;
    }

    const requests = group.variants.map((variant) =>
      fetch(`/api/districts/?state=${encodeURIComponent(variant)}`)
        .then((res) => {
          if (!res.ok) throw new Error('Failed to load districts.');
          return res.json();
        })
        .then((data) => {
          const list = Array.isArray(data) ? data : data.districts;
          return (list || []).map((d) => ({ name: String(d).trim(), variant }));
        })
        .catch(() => [])
    );

    Promise.all(requests)
      .then((results) => {
        const merged = [];
        const seen = new Set();

        results.flat().forEach(({ name, variant }) => {
          if (!name) return;
          const key = name.toUpperCase();
          if (seen.has(key)) return;
          seen.add(key);
          merged.push(name);
          districtStateMap[name] = variant;
        });

        if (merged.length === 0) {
          districtSelect.innerHTML = '<option value="">No districts found</option>';
          showError('No districts were found for the selected state.');
          return;
        }

        merged.sort((a, b) => a.localeCompare(b));

        districtSelect.innerHTML = '<option value="">Select District</option>';
        merged.forEach((district) => {
          const opt = document.createElement('option');
          opt.value = district;
          opt.textContent = district;
          districtSelect.appendChild(opt);
        });
        districtSelect.disabled = false;
      })
      .catch(() => {
        districtSelect.innerHTML = '<option value="">Unable to load districts</option>';
        showError('Could not load districts for the selected state. Please try again.');
      });
  }

  stateSelect.addEventListener('change', () => {
    clearError();
    const key = stateSelect.value;
    if (key) {
      loadDistricts(key);
    } else {
      districtSelect.disabled = true;
      districtSelect.innerHTML = '<option value="">Select State First</option>';
      districtStateMap = {};
    }
  });

  // -------------------------------------------------------------------
  // Risk styling helper
  // -------------------------------------------------------------------
  function riskClass(riskLevel) {
    switch ((riskLevel || '').toLowerCase()) {
      case 'low':
        return 'risk-text-low';
      case 'moderate':
        return 'risk-text-moderate';
      case 'high':
        return 'risk-text-high';
      case 'very high':
        return 'risk-text-very-high';
      default:
        return '';
    }
  }

  // -------------------------------------------------------------------
  // Theme-aware chart colors
  // -------------------------------------------------------------------
  function getChartColors() {
    const styles = getComputedStyle(document.body);
    const read = (name, fallback) => {
      const value = styles.getPropertyValue(name).trim();
      return value || fallback;
    };
    return {
      text: read('--fc-navy', '#24344D'),
      muted: read('--fc-text-muted', '#666666'),
      grid: read('--fc-chart-grid', 'rgba(36, 52, 77, 0.08)'),
      historical: '#6E86AB',
      historicalFill: 'rgba(110, 134, 171, 0.12)',
      forecast: '#D0554A',
    };
  }

  // -------------------------------------------------------------------
  // Chart rendering (Historical Trend + Forecast)
  // -------------------------------------------------------------------
  let forecastChartInstance = null;
  let lastChartData = null;

  const forecastMarkerPlugin = {
    id: 'forecastMarkerPlugin',
    afterDatasetsDraw(chart) {
      if (!lastChartData) return;
      const { ctx, chartArea, scales } = chart;
      const xScale = scales.x;
      const forecastIndex = lastChartData.forecastIndex;
      if (forecastIndex === undefined || !xScale) return;

      const x = xScale.getPixelForValue(forecastIndex);
      const colors = getChartColors();

      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = colors.forecast;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors.forecast;
      ctx.font = '600 11px "Segoe UI", Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Forecast', x, chartArea.top + 14);
      ctx.restore();
    },
  };

  function renderForecastChart(historicalTrend, forecastYear, predictedCount) {
    if (!chartCanvas || typeof Chart === 'undefined') return;

    const historyLabels = (historicalTrend || []).map((item) => item.year);
    const historyValues = (historicalTrend || []).map((item) => item.crime_count);

    const labels = [...historyLabels, forecastYear];
    const forecastIndex = labels.length - 1;

    const historicalSeries = [...historyValues, null];

    const forecastSeries = historyValues.map(() => null);
    if (historyValues.length > 0) {
      forecastSeries[forecastSeries.length - 1] = historyValues[historyValues.length - 1];
    }
    forecastSeries.push(predictedCount);

    // Calculate dynamic extrapolation uncertainty envelope
    const yearsAhead = Math.max(1, (Number(forecastYear) || 2026) - 2013);
    const uncertaintyRatio = Math.min(0.40, 0.06 + yearsAhead * 0.018);
    const lastHistVal = historyValues.length > 0 ? historyValues[historyValues.length - 1] : predictedCount;

    const upperSeries = historyValues.map(() => null);
    const lowerSeries = historyValues.map(() => null);
    if (historyValues.length > 0) {
      upperSeries[upperSeries.length - 1] = lastHistVal;
      lowerSeries[lowerSeries.length - 1] = lastHistVal;
    }
    upperSeries.push(Math.round(predictedCount * (1 + uncertaintyRatio)));
    lowerSeries.push(Math.max(0, Math.round(predictedCount * (1 - uncertaintyRatio))));

    lastChartData = { labels, historicalSeries, forecastSeries, upperSeries, lowerSeries, forecastIndex, showFan: true };

    if (forecastChartInstance) {
      forecastChartInstance.destroy();
    }

    const colors = getChartColors();

    const datasets = [
      {
        label: 'Historical Crime Trend',
        data: historicalSeries,
        borderColor: colors.historical,
        backgroundColor: colors.historicalFill,
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        pointBackgroundColor: colors.historical,
        spanGaps: false,
      },
      {
        label: 'Future Forecast',
        data: forecastSeries,
        borderColor: colors.forecast,
        backgroundColor: 'transparent',
        borderDash: [6, 6],
        fill: false,
        tension: 0,
        pointRadius: (ctx) => (ctx.dataIndex === forecastIndex ? 7 : 0),
        pointHoverRadius: (ctx) => (ctx.dataIndex === forecastIndex ? 9 : 0),
        pointBackgroundColor: colors.forecast,
        pointBorderColor: '#FFFFFF',
        pointBorderWidth: 2,
        spanGaps: true,
      }
    ];

    if (lastChartData.showFan) {
      datasets.push({
        label: 'Upper Confidence Bound (+Uncertainty)',
        data: upperSeries,
        borderColor: 'rgba(208, 85, 74, 0.4)',
        backgroundColor: 'rgba(208, 85, 74, 0.12)',
        borderDash: [2, 2],
        borderWidth: 1,
        fill: '+1',
        pointRadius: 0,
        spanGaps: true,
      });
      datasets.push({
        label: 'Lower Confidence Bound (-Uncertainty)',
        data: lowerSeries,
        borderColor: 'rgba(208, 85, 74, 0.4)',
        backgroundColor: 'transparent',
        borderDash: [2, 2],
        borderWidth: 1,
        fill: false,
        pointRadius: 0,
        spanGaps: true,
      });
    }

    forecastChartInstance = new Chart(chartCanvas, {
      type: 'line',
      data: {
        labels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 900, easing: 'easeOutQuart' },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: { color: colors.text },
          },
          tooltip: { enabled: true },
        },
        scales: {
          y: {
            beginAtZero: false,
            grid: { color: colors.grid },
            ticks: { color: colors.muted },
            title: { display: true, text: 'Crime Count', color: colors.muted },
          },
          x: {
            grid: { display: false },
            ticks: { color: colors.muted },
            title: { display: true, text: 'Year', color: colors.muted },
          },
        },
      },
      plugins: [forecastMarkerPlugin],
    });

    showChart();
  }

  function refreshChartTheme() {
    if (!forecastChartInstance || !lastChartData) return;
    const colors = getChartColors();

    forecastChartInstance.data.datasets[0].borderColor = colors.historical;
    forecastChartInstance.data.datasets[0].backgroundColor = colors.historicalFill;
    forecastChartInstance.data.datasets[0].pointBackgroundColor = colors.historical;
    forecastChartInstance.data.datasets[1].borderColor = colors.forecast;
    forecastChartInstance.data.datasets[1].pointBackgroundColor = colors.forecast;

    forecastChartInstance.options.plugins.legend.labels.color = colors.text;
    forecastChartInstance.options.scales.y.grid.color = colors.grid;
    forecastChartInstance.options.scales.y.ticks.color = colors.muted;
    forecastChartInstance.options.scales.y.title.color = colors.muted;
    forecastChartInstance.options.scales.x.ticks.color = colors.muted;
    forecastChartInstance.options.scales.x.title.color = colors.muted;

    forecastChartInstance.update();
  }

  const themeObserver = new MutationObserver(() => {
    window.requestAnimationFrame(refreshChartTheme);
  });
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class', 'data-theme', 'data-bs-theme'],
  });
  themeObserver.observe(document.body, {
    attributes: true,
    attributeFilter: ['class', 'data-theme', 'data-bs-theme'],
  });

  // -------------------------------------------------------------------
  // Run Forecast Handler
  // -------------------------------------------------------------------
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    clearError();

    const stateKey = stateSelect.value;
    const stateGroup = stateGroups[stateKey];
    const districtValue = districtSelect.value;
    const year = yearSelect.value;

    if (!stateGroup) {
      showError('Please select a state.');
      return;
    }
    if (!districtValue) {
      showError('Please select a district.');
      return;
    }
    if (!year) {
      showError('Please select a forecast year.');
      return;
    }

    const backendState = districtStateMap[districtValue] || stateGroup.variants[0];

    setLoading(true);

    const predictRequest = fetch('/api/predict/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken,
      },
      body: JSON.stringify({
        state: backendState,
        district: districtValue,
        year: year,
      }),
    }).then(async (res) => {
      let payload;
      try {
        payload = await res.json();
      } catch (e) {
        throw new Error('The server returned an unexpected response.');
      }
      if (!res.ok || payload.status === 'error') {
        throw new Error(payload.message || 'Unable to generate a forecast for this selection.');
      }
      return payload;
    });

    const trendRequest = fetch(
      `/api/forecast-trend/?state=${encodeURIComponent(backendState)}&district=${encodeURIComponent(districtValue)}`
    )
      .then(async (res) => {
        let payload;
        try {
          payload = await res.json();
        } catch (e) {
          return { status: 'error' };
        }
        if (!res.ok || payload.status === 'error') {
          return { status: 'error' };
        }
        return payload;
      })
      .catch(() => ({ status: 'error' }));

    Promise.all([predictRequest, trendRequest])
      .then(([predictPayload, trendPayload]) => {
        document.getElementById('fcState').textContent = stateGroup.display;
        document.getElementById('fcDistrict').textContent = districtValue;
        document.getElementById('fcYear').textContent = year;
        document.getElementById('fcModel').textContent = predictPayload.model || 'Hybrid CNN-LSTM';

        document.getElementById('fcPredictedCount').textContent =
          predictPayload.predicted_count !== undefined && predictPayload.predicted_count !== null
            ? Math.round(predictPayload.predicted_count).toLocaleString()
            : '—';

        const riskLevelEl = document.getElementById('fcRiskLevel');
        riskLevelEl.textContent = predictPayload.risk_level ?? '—';
        riskLevelEl.className = 'result-value ' + riskClass(predictPayload.risk_level);

        document.getElementById('fcRiskScore').textContent =
          predictPayload.risk_score !== undefined && predictPayload.risk_score !== null
            ? predictPayload.risk_score
            : '—';

        document.getElementById('fcConfidence').textContent =
          predictPayload.confidence !== undefined && predictPayload.confidence !== null
            ? `${Math.round(predictPayload.confidence * 10) / 10}%`
            : '—';

        document.getElementById('fcRecommendation').textContent =
          predictPayload.recommendation || 'No recommendation available.';

        if (resultsSection) {
          resultsSection.classList.remove('d-none');
        }

        // Map backend `{ status: "success", years: [...], values: [...] }` to historical array
        if (
          trendPayload.status === 'success' &&
          Array.isArray(trendPayload.years) &&
          Array.isArray(trendPayload.values) &&
          trendPayload.years.length > 0 &&
          trendPayload.values.length > 0
        ) {
          const historicalTrend = trendPayload.years.map((yearValue, index) => ({
            year: Number(yearValue),
            crime_count: Number(trendPayload.values[index]),
          }));

          renderForecastChart(
            historicalTrend,
            Number(year),
            Number(predictPayload.predicted_count)
          );
        } else {
          showChartEmptyState();
        }

        if (resultsSection) {
          resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      })
      .catch((err) => {
        showChartEmptyState();
        showError(err.message || 'Something went wrong while generating the forecast. Please try again.');
      })
      .finally(() => {
        setLoading(false);
      });
  });

  const btnFanOn = document.getElementById('btnFanOn');
  const btnFanOff = document.getElementById('btnFanOff');
  if (btnFanOn && btnFanOff) {
    btnFanOn.addEventListener('click', () => {
      btnFanOn.classList.add('active');
      btnFanOff.classList.remove('active');
      if (forecastChartInstance) {
        forecastChartInstance.data.datasets.forEach((ds) => {
          if (ds.label && ds.label.includes('Confidence')) ds.hidden = false;
        });
        forecastChartInstance.update();
      }
    });
    btnFanOff.addEventListener('click', () => {
      btnFanOff.classList.add('active');
      btnFanOn.classList.remove('active');
      if (forecastChartInstance) {
        forecastChartInstance.data.datasets.forEach((ds) => {
          if (ds.label && ds.label.includes('Confidence')) ds.hidden = true;
        });
        forecastChartInstance.update();
      }
    });
  }

  populateYears();
  loadStates();
  showChartEmptyState();
});