/**
 * performance.js
 * ---------------------------------------------------------------------------
 * Renders the Performance page charts using Chart.js v4:
 *   1. R² / Accuracy Comparison -> #accuracyChart (bar chart)
 *   2. Training vs Validation Loss -> #lossChart (line chart, only if the
 *      backend actually found results/hybrid_training_history.csv)
 *
 * All numeric data comes from the <script id="performance-chart-data"
 * type="application/json"> tag that performance.html renders from the
 * Django view's context (see views.py: MODEL_PERFORMANCE_METRICS /
 * performance()). Nothing in this file is hardcoded, so the chart can
 * never drift out of sync with the KPI cards or the evaluation table
 * again.
 *
 * Colors for text/gridlines are read from the page's CSS custom
 * properties at render time (not hardcoded hex), so both charts stay
 * readable in light mode and dark mode, and update automatically when
 * the theme toggle flips `data-theme` on <html>/<body>.
 * ---------------------------------------------------------------------------
 */

document.addEventListener('DOMContentLoaded', () => {
  if (typeof Chart === 'undefined') {
    console.error(
      '[performance.js] Chart.js is not loaded. ' +
      'Add the Chart.js <script> tag before performance.js.'
    );
    return;
  }

  const dataEl = document.getElementById('performance-chart-data');
  if (!dataEl) {
    console.error('[performance.js] #performance-chart-data script tag not found.');
    return;
  }

  let payload;
  try {
    payload = JSON.parse(dataEl.textContent);
  } catch (err) {
    console.error('[performance.js] Could not parse performance chart data:', err);
    return;
  }

  // Distinct accent colors per model series. The site's locked palette
  // only defines two brand colors (primary/secondary), which isn't
  // enough to tell three bars apart, so these three chart-only accents
  // are used purely for series identification - text, gridlines and
  // legends still come from the theme variables below.
  const SERIES_COLOR = {
    cnn: '#4C6EF5',
    lstm: '#12B886',
    hybrid: '#F59F00',
  };
  const SERIES_BORDER = {
    cnn: '#3B5BDB',
    lstm: '#0CA678',
    hybrid: '#E8590C',
  };

  const SMOOTH_ANIMATION = { duration: 900, easing: 'easeOutQuart' };

  /**
   * Reads the current theme's text/border colors straight from the
   * page's CSS custom properties, so charts follow light/dark mode
   * instead of using a fixed color that can vanish on one of them.
   */
  function getThemeColors() {
    const styles = getComputedStyle(document.documentElement);
    const read = (name, fallback) => (styles.getPropertyValue(name) || fallback).trim();
    return {
      text: read('--color-heading', '#24344D'),
      muted: read('--color-paragraph', '#666666'),
      grid: read('--color-border', '#E5E5E5'),
    };
  }

  let accuracyChartInstance = null;
  let lossChartInstance = null;

  function buildBarValueLabelPlugin(textColor) {
    return {
      id: 'barValueLabelPlugin',
      afterDatasetsDraw(chart) {
        const { ctx } = chart;
        ctx.save();
        ctx.font = '600 12px "Inter", "Segoe UI", Arial, sans-serif';
        ctx.fillStyle = textColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';

        chart.data.datasets.forEach((dataset, datasetIndex) => {
          const meta = chart.getDatasetMeta(datasetIndex);
          if (meta.hidden) return;
          meta.data.forEach((bar, index) => {
            const value = dataset.data[index];
            if (value === null || value === undefined) return;
            ctx.fillText(`${value}%`, bar.x, bar.y - 6);
          });
        });

        ctx.restore();
      },
    };
  }

  function renderAccuracyChart() {
    const canvas = document.getElementById('accuracyChart');
    if (!canvas) return;

    if (accuracyChartInstance) {
      accuracyChartInstance.destroy();
      accuracyChartInstance = null;
    }

    const theme = getThemeColors();
    const labels = payload.labels || [];
    const values = payload.accuracy || [];
    const barColors = [SERIES_COLOR.cnn, SERIES_COLOR.lstm, SERIES_COLOR.hybrid];
    const borderColors = [SERIES_BORDER.cnn, SERIES_BORDER.lstm, SERIES_BORDER.hybrid];

    accuracyChartInstance = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'R² Score (%)',
            data: values,
            backgroundColor: barColors.slice(0, labels.length),
            borderColor: borderColors.slice(0, labels.length),
            borderWidth: 1.5,
            borderRadius: 6,
            maxBarThickness: 70,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 24 } },
        animation: SMOOTH_ANIMATION,
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: theme.text } },
          tooltip: {
            enabled: true,
            callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%` },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            suggestedMax: 100,
            grid: { color: theme.grid },
            ticks: { color: theme.muted, callback: (value) => `${value}%` },
          },
          x: {
            grid: { display: false },
            ticks: { color: theme.text },
          },
        },
      },
      plugins: [buildBarValueLabelPlugin(theme.text)],
    });
  }

  function renderLossChart() {
    const canvas = document.getElementById('lossChart');
    const emptyNote = document.getElementById('lossChartEmpty');
    if (!canvas) return;

    const history = payload.loss_history;
    const hasHistory =
      history &&
      Array.isArray(history.training_loss) &&
      Array.isArray(history.validation_loss) &&
      history.training_loss.length > 0;

    if (!hasHistory) {
      canvas.hidden = true;
      if (emptyNote) emptyNote.hidden = false;
      return;
    }

    canvas.hidden = false;
    if (emptyNote) emptyNote.hidden = true;

    if (lossChartInstance) {
      lossChartInstance.destroy();
      lossChartInstance = null;
    }

    const theme = getThemeColors();

    lossChartInstance = new Chart(canvas, {
      type: 'line',
      data: {
        labels: history.epochs,
        datasets: [
          {
            label: 'Training Loss',
            data: history.training_loss,
            borderColor: SERIES_COLOR.cnn,
            backgroundColor: 'rgba(76, 110, 245, 0.15)',
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointHoverRadius: 5,
          },
          {
            label: 'Validation Loss',
            data: history.validation_loss,
            borderColor: SERIES_COLOR.hybrid,
            backgroundColor: 'rgba(245, 159, 0, 0.12)',
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointHoverRadius: 5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: SMOOTH_ANIMATION,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: theme.text } },
          tooltip: { enabled: true },
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: theme.grid },
            ticks: { color: theme.muted },
            title: { display: true, text: 'Loss (MSE)', color: theme.text },
          },
          x: {
            grid: { display: false },
            ticks: { color: theme.text },
            title: { display: true, text: 'Epoch', color: theme.text },
          },
        },
      },
    });
  }

  // =========================================================================
  // 5-FOLD CROSS-VALIDATION COMPARATOR & SCATTER PLOT
  // =========================================================================
  let evalDataCache = null;
  let cvChartInstance = null;
  let scatterChartInstance = null;
  let currentCVMetric = 'r2';

  function getEmbeddedCVData() {
    const el = document.getElementById('cv-benchmark-data');
    if (!el || !el.textContent.trim()) return null;
    try {
      const raw = JSON.parse(el.textContent);
      if (!raw || !raw.per_fold) return null;
      
      const folds = [];
      ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'].forEach((name, i) => {
        if (raw.per_fold[name]) {
          folds.push({
            fold: i + 1,
            metrics_2way: raw.per_fold[name]['2way'],
            metrics_3way: raw.per_fold[name]['3way_gcn']
          });
        }
      });

      const agg = raw.aggregate || {};
      const summary = {
        mean_2way: {
          r2: agg.r2 ? agg.r2['2way_mean'] : (folds.reduce((sum, f) => sum + (f.metrics_2way.r2 || 0), 0) / (folds.length || 1)),
          rmse: agg.rmse ? agg.rmse['2way_mean'] : (folds.reduce((sum, f) => sum + (f.metrics_2way.rmse || 0), 0) / (folds.length || 1)),
          mae: agg.mae ? agg.mae['2way_mean'] : (folds.reduce((sum, f) => sum + (f.metrics_2way.mae || 0), 0) / (folds.length || 1)),
          medae: agg.medae ? agg.medae['2way_mean'] : (folds.reduce((sum, f) => sum + (f.metrics_2way.medae || 0), 0) / (folds.length || 1))
        },
        mean_3way: {
          r2: agg.r2 ? agg.r2['3way_mean'] : (folds.reduce((sum, f) => sum + (f.metrics_3way.r2 || 0), 0) / (folds.length || 1)),
          rmse: agg.rmse ? agg.rmse['3way_mean'] : (folds.reduce((sum, f) => sum + (f.metrics_3way.rmse || 0), 0) / (folds.length || 1)),
          mae: agg.mae ? agg.mae['3way_mean'] : (folds.reduce((sum, f) => sum + (f.metrics_3way.mae || 0), 0) / (folds.length || 1)),
          medae: agg.medae ? agg.medae['3way_mean'] : (folds.reduce((sum, f) => sum + (f.metrics_3way.medae || 0), 0) / (folds.length || 1))
        }
      };

      return { cv_folds: folds, summary: summary };
    } catch (e) {
      console.warn('[performance.js] Error parsing embedded CV data:', e);
      return null;
    }
  }

  async function loadEvalData() {
    if (evalDataCache) return evalDataCache;

    // Check embedded script tag first
    const embedded = getEmbeddedCVData();
    if (embedded) {
      evalDataCache = embedded;
    }

    try {
      const res = await fetch('/static/data/eval_scatter_data.json');
      if (res.ok) {
        const fetched = await res.json();
        if (evalDataCache) {
          evalDataCache.scatter_points = fetched.scatter_points || [];
          if (!evalDataCache.cv_folds || evalDataCache.cv_folds.length === 0) {
            evalDataCache.cv_folds = fetched.cv_folds || [];
            evalDataCache.summary = fetched.summary || {};
          }
        } else {
          evalDataCache = fetched;
        }
      }
    } catch (err) {
      console.error('[performance.js] Error loading eval scatter data:', err);
    }
    return evalDataCache;
  }

  async function renderCVFoldChart(metric = currentCVMetric) {
    const canvas = document.getElementById('cvFoldChart');
    if (!canvas) return;
    const theme = getThemeColors();
    const data = await loadEvalData();
    if (!data || !data.cv_folds || data.cv_folds.length === 0) return;

    currentCVMetric = metric;
    const folds = data.cv_folds;
    const labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean (Overall)'];

    const vals2Way = folds.map(f => {
      if (!f.metrics_2way) return 0;
      if (metric === 'r2') return (f.metrics_2way.r2 || 0) * 100;
      if (metric === 'rmse') return f.metrics_2way.rmse || 0;
      return f.metrics_2way.mae || 0;
    });

    const vals3Way = folds.map(f => {
      if (!f.metrics_3way) return 0;
      if (metric === 'r2') return (f.metrics_3way.r2 || 0) * 100;
      if (metric === 'rmse') return f.metrics_3way.rmse || 0;
      return f.metrics_3way.mae || 0;
    });

    // Append summary mean
    if (data.summary) {
      if (data.summary.mean_2way) {
        vals2Way.push(metric === 'r2' ? (data.summary.mean_2way.r2 || 0) * 100 : (metric === 'rmse' ? (data.summary.mean_2way.rmse || 0) : (data.summary.mean_2way.mae || 0)));
      }
      if (data.summary.mean_3way) {
        vals3Way.push(metric === 'r2' ? (data.summary.mean_3way.r2 || 0) * 100 : (metric === 'rmse' ? (data.summary.mean_3way.rmse || 0) : (data.summary.mean_3way.mae || 0)));
      }
    }

    if (cvChartInstance) cvChartInstance.destroy();

    cvChartInstance = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: '2-Way Hybrid (CNN-LSTM)',
            data: vals2Way,
            backgroundColor: 'rgba(76, 110, 245, 0.75)',
            borderColor: '#4C6EF5',
            borderWidth: 1.5,
            borderRadius: 4
          },
          {
            label: '3-Way Hybrid (CNN-LSTM-GCN)',
            data: vals3Way,
            backgroundColor: 'rgba(245, 159, 0, 0.85)',
            borderColor: '#F59F00',
            borderWidth: 1.5,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: SMOOTH_ANIMATION,
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: theme.text, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}${metric === 'r2' ? '%' : ''}`
            }
          }
        },
        scales: {
          y: {
            grid: { color: theme.grid },
            ticks: { color: theme.muted, callback: (v) => `${v}${metric === 'r2' ? '%' : ''}` },
            title: { display: true, text: metric === 'r2' ? 'R² Score (%)' : metric.toUpperCase(), color: theme.text }
          },
          x: {
            grid: { display: false },
            ticks: { color: theme.text }
          }
        }
      }
    });
  }

  function getEmbeddedScatterPoints() {
    const el = document.getElementById('scatter-points-data');
    if (!el || !el.textContent.trim()) return null;
    try {
      const raw = JSON.parse(el.textContent);
      if (Array.isArray(raw) && raw.length > 0) return raw;
      return null;
    } catch (e) {
      console.warn('[performance.js] Error parsing embedded scatter data:', e);
      return null;
    }
  }

  async function renderScatterChart() {
    const canvas = document.getElementById('scatterChart');
    if (!canvas) return;
    const theme = getThemeColors();
    
    let points = getEmbeddedScatterPoints();
    if (!points || points.length === 0) {
      const data = await loadEvalData();
      points = (data && data.scatter_points) ? data.scatter_points : [];
    }

    if (!points || points.length === 0) {
      console.warn('[performance.js] No scatter points available.');
      return;
    }

    const scatter3Way = points.map(p => ({ x: p.actual, y: p.predicted_3way, meta: p }));

    // Find max value for 45-degree reference line
    const maxVal = Math.max(...points.map(p => Math.max(p.actual, p.predicted_3way)), 5000);

    if (scatterChartInstance) scatterChartInstance.destroy();

    scatterChartInstance = new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Ideal Prediction (y = x)',
            data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
            type: 'line',
            showLine: true,
            borderColor: '#94A3B8',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          },
          {
            label: '3-Way Predictions (Held-Out Test)',
            data: scatter3Way,
            backgroundColor: 'rgba(245, 159, 0, 0.75)',
            borderColor: '#D97706',
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: SMOOTH_ANIMATION,
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: theme.text, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const p = ctx.raw.meta;
                if (!p) return 'Ideal fit line (y = x)';
                return `${p.district} (${p.state}, ${p.year}): Actual = ${p.actual.toLocaleString()}, Pred = ${p.predicted_3way.toLocaleString()} (Err: ${p.residual_3way > 0 ? '+' : ''}${p.residual_3way})`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: theme.grid },
            ticks: { color: theme.muted },
            title: { display: true, text: 'Actual Crime Count', color: theme.text }
          },
          y: {
            grid: { color: theme.grid },
            ticks: { color: theme.muted },
            title: { display: true, text: 'Predicted Crime Count', color: theme.text }
          }
        }
      }
    });
  }

  function bindCVControls() {
    const group = document.getElementById('cvMetricBtnGroup');
    if (!group) return;
    group.querySelectorAll('button[data-metric]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        group.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const metric = e.target.getAttribute('data-metric');
        renderCVFoldChart(metric);
      });
    });
  }

  function renderAll() {
    renderAccuracyChart();
    renderLossChart();
    renderCVFoldChart(currentCVMetric);
    renderScatterChart();
  }

  bindCVControls();
  renderAll();

  // Re-render with fresh theme colors on dark-mode toggle
  const themeTargets = [document.documentElement, document.body].filter(Boolean);
  const themeObserver = new MutationObserver(() => renderAll());
  themeTargets.forEach((el) => {
    themeObserver.observe(el, { attributes: true, attributeFilter: ['data-theme'] });
  });
});