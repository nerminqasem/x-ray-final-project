/* =========================================================
   PneumoVision – charts.js
   Centralized chart rendering for all pages.
   Requires: Chart.js (loaded in HTML via CDN)
   ========================================================= */

(function () {
    'use strict';

    /* ---------------------------------------------------------
       Global Chart.js defaults (dark premium theme)
       --------------------------------------------------------- */
    if (window.Chart) {
        Chart.defaults.color = '#a0a0b8';
        Chart.defaults.borderColor = '#2a2a45';
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.font.size = 12;
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        Chart.defaults.plugins.legend.labels.padding = 16;
        Chart.defaults.plugins.tooltip.backgroundColor = '#1c1c30';
        Chart.defaults.plugins.tooltip.borderColor = '#2a2a45';
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.titleColor = '#e8e8f0';
        Chart.defaults.plugins.tooltip.bodyColor = '#e8e8f0';
        Chart.defaults.plugins.tooltip.padding = 10;
        Chart.defaults.plugins.tooltip.cornerRadius = 8;
    }

    /* ---------------------------------------------------------
       Color palette
       --------------------------------------------------------- */
    const COLORS = {
        primary:  'rgba(156, 39, 176, 0.75)',
        primaryB: 'rgba(156, 39, 176, 1)',
        accent:   'rgba(0, 188, 212, 0.75)',
        accentB:  'rgba(0, 188, 212, 1)',
        success:  'rgba(76, 175, 80, 0.75)',
        successB: 'rgba(76, 175, 80, 1)',
        danger:   'rgba(244, 67, 54, 0.75)',
        dangerB:  'rgba(244, 67, 54, 1)',
        warning:  'rgba(255, 152, 0, 0.75)',
        warningB: 'rgba(255, 152, 0, 1)',
        muted:    'rgba(160, 160, 184, 0.6)',
    };

    const PALETTE = [
        COLORS.primary,
        COLORS.accent,
        COLORS.success,
        COLORS.warning,
        COLORS.danger,
    ];

    /* ---------------------------------------------------------
       Helpers
       --------------------------------------------------------- */
    function destroyIfExists(id) {
        const existing = Chart.getChart(id);
        if (existing) existing.destroy();
    }

    function toPercent(v) {
        if (v === null || v === undefined) return 0;
        return v <= 1 ? v * 100 : v;
    }

    function fmtPct(v) {
        return toPercent(v).toFixed(2) + '%';
    }

    /* =========================================================
       1) MODEL COMPARISON – Grouped Bar Chart
       ========================================================= */
    function renderModelComparison(canvasId, models) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !models) return;

        const names = Object.keys(models);
        const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'];
        const labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'];

        const datasets = names.map((name, i) => ({
            label: name,
            data: metrics.map(m => toPercent(models[name][m])),
            backgroundColor: PALETTE[i % PALETTE.length],
            borderColor: PALETTE[i % PALETTE.length].replace('0.75', '1'),
            borderWidth: 1.5,
            borderRadius: 6,
            barPercentage: 0.7,
            categoryPercentage: 0.65,
        }));

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'bar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: v => v + '%' },
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    /* =========================================================
       2) PREPROCESSING COMPARISON – Line Chart
       ========================================================= */
    function renderPreprocessingComparison(canvasId, rows) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !rows || !rows.length) return;

        const labels = rows.map(r => r.preprocessing);
        const datasets = [
            {
                label: 'Accuracy',
                data: rows.map(r => toPercent(r.accuracy)),
                borderColor: COLORS.primaryB,
                backgroundColor: COLORS.primary,
                tension: 0.35,
                fill: false,
                pointRadius: 5,
                pointHoverRadius: 7,
            },
            {
                label: 'F1-Score',
                data: rows.map(r => toPercent(r.f1_score)),
                borderColor: COLORS.accentB,
                backgroundColor: COLORS.accent,
                tension: 0.35,
                fill: false,
                pointRadius: 5,
                pointHoverRadius: 7,
            },
            {
                label: 'ROC-AUC',
                data: rows.map(r => toPercent(r.roc_auc)),
                borderColor: COLORS.successB,
                backgroundColor: COLORS.success,
                tension: 0.35,
                fill: false,
                pointRadius: 5,
                pointHoverRadius: 7,
            }
        ];

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: v => v + '%' },
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    /* =========================================================
       3) DATASET SPLIT – Bar Chart
       ========================================================= */
    function renderDatasetSplit(canvasId, stats) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !stats) return;

        const labels = ['Train', 'Validation', 'Test'];
        const values = [
            stats.train ? stats.train.total : 0,
            stats.val   ? stats.val.total   : 0,
            stats.test  ? stats.test.total  : 0,
        ];

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Total Images',
                    data: values,
                    backgroundColor: [COLORS.primary, COLORS.accent, COLORS.success],
                    borderRadius: 8,
                    barPercentage: 0.55,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: { label: ctx => ` ${ctx.parsed.y} images` }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    /* =========================================================
       4) CLASS DISTRIBUTION – Doughnut Chart
       ========================================================= */
    function renderClassDistribution(canvasId, stats) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !stats) return;

        const normal =
            (stats.train?.normal || 0) +
            (stats.val?.normal   || 0) +
            (stats.test?.normal  || 0);

        const pneumonia =
            (stats.train?.pneumonia || 0) +
            (stats.val?.pneumonia   || 0) +
            (stats.test?.pneumonia  || 0);

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels: ['Normal', 'Pneumonia'],
                datasets: [{
                    data: [normal, pneumonia],
                    backgroundColor: [COLORS.success, COLORS.danger],
                    borderColor: '#1c1c30',
                    borderWidth: 3,
                    hoverOffset: 10,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '62%',
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const total = normal + pneumonia;
                                const pct = total ? ((ctx.parsed / total) * 100).toFixed(2) : 0;
                                return ` ${ctx.label}: ${ctx.parsed} (${pct}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    /* =========================================================
       5) CONFUSION MATRIX – Heatmap-style Bar
       ========================================================= */
    function renderConfusionMatrix(canvasId, cm) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !cm) return;

        // cm = [[TN, FP], [FN, TP]]
        const labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive'];
        const values = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]];
        const colors = [COLORS.success, COLORS.warning, COLORS.warning, COLORS.success];

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Count',
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 8,
                    barPercentage: 0.6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y} images` } }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(42, 42, 69, 0.5)' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    /* =========================================================
       6) ROC CURVE – Line Chart
       ========================================================= */
    function renderROCCurve(canvasId, curves) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !curves) return;

        // curves = { "DenseNet121": {fpr:[], tpr:[]}, "MobileNetV3": {...} }
        const datasets = [];
        let i = 0;
        for (const [name, data] of Object.entries(curves)) {
            datasets.push({
                label: name,
                data: data.fpr.map((x, idx) => ({ x, y: data.tpr[idx] })),
                borderColor: PALETTE[i % PALETTE.length].replace('0.75', '1'),
                backgroundColor: PALETTE[i % PALETTE.length],
                borderWidth: 2.5,
                tension: 0.25,
                pointRadius: 3,
                fill: false,
            });
            i++;
        }

        // Diagonal reference line
        datasets.push({
            label: 'Random',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: 'rgba(160, 160, 184, 0.6)',
            borderDash: [6, 6],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
        });

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                parsing: false,
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: {
                        callbacks: {
                            label: ctx => `${ctx.dataset.label} (FPR=${ctx.parsed.x.toFixed(2)}, TPR=${ctx.parsed.y.toFixed(2)})`
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0, max: 1,
                        title: { display: true, text: 'False Positive Rate' },
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    },
                    y: {
                        min: 0, max: 1,
                        title: { display: true, text: 'True Positive Rate' },
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    }
                }
            }
        });
    }

    /* =========================================================
       7) TRAINING LOSS – Line Chart
       ========================================================= */
    function renderTrainingLoss(canvasId, histories) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !histories) return;

        // histories = { "DenseNet121": {train_loss:[], val_loss:[]}, ... }
        const datasets = [];
        let i = 0;

        for (const [name, h] of Object.entries(histories)) {
            const epochs = h.train_loss.map((_, idx) => idx + 1);

            datasets.push({
                label: `${name} – Train Loss`,
                data: epochs.map((e, idx) => ({ x: e, y: h.train_loss[idx] })),
                borderColor: PALETTE[i % PALETTE.length].replace('0.75', '1'),
                borderWidth: 2,
                tension: 0.35,
                pointRadius: 0,
                fill: false,
            });
            datasets.push({
                label: `${name} – Val Loss`,
                data: epochs.map((e, idx) => ({ x: e, y: h.val_loss[idx] })),
                borderColor: PALETTE[i % PALETTE.length].replace('0.75', '1'),
                borderDash: [5, 5],
                borderWidth: 2,
                tension: 0.35,
                pointRadius: 0,
                fill: false,
            });
            i++;
        }

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                parsing: false,
                plugins: { legend: { position: 'bottom' } },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Epoch' },
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    },
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Loss' },
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    }
                }
            }
        });
    }

    /* =========================================================
       8) TRAINING ACCURACY – Line Chart
       ========================================================= */
    function renderTrainingAccuracy(canvasId, histories) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !histories) return;

        const datasets = [];
        let i = 0;

        for (const [name, h] of Object.entries(histories)) {
            const epochs = h.train_acc.map((_, idx) => idx + 1);

            datasets.push({
                label: `${name} – Train Acc`,
                data: epochs.map((e, idx) => ({ x: e, y: toPercent(h.train_acc[idx]) })),
                borderColor: PALETTE[i % PALETTE.length].replace('0.75', '1'),
                borderWidth: 2,
                tension: 0.35,
                pointRadius: 0,
                fill: false,
            });
            datasets.push({
                label: `${name} – Val Acc`,
                data: epochs.map((e, idx) => ({ x: e, y: toPercent(h.val_acc[idx]) })),
                borderColor: PALETTE[i % PALETTE.length].replace('0.75', '1'),
                borderDash: [5, 5],
                borderWidth: 2,
                tension: 0.35,
                pointRadius: 0,
                fill: false,
            });
            i++;
        }

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                parsing: false,
                plugins: { legend: { position: 'bottom' } },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Epoch' },
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Accuracy (%)' },
                        ticks: { callback: v => v + '%' },
                        grid: { color: 'rgba(42, 42, 69, 0.5)' }
                    }
                }
            }
        });
    }

    /* =========================================================
       9) PREDICTION DISTRIBUTION – Pie Chart (History)
       ========================================================= */
    function renderPredictionDistribution(canvasId, counts) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !counts) return;

        // counts = { NORMAL: n, PNEUMONIA: m }
        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'pie',
            data: {
                labels: ['Normal', 'Pneumonia'],
                datasets: [{
                    data: [counts.NORMAL || 0, counts.PNEUMONIA || 0],
                    backgroundColor: [COLORS.success, COLORS.danger],
                    borderColor: '#1c1c30',
                    borderWidth: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    /* =========================================================
       10) MODEL ACCURACY RADAR – Radar Chart
       ========================================================= */
    function renderModelRadar(canvasId, models) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !models) return;

        const names = Object.keys(models);
        const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'];
        const labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'];

        const datasets = names.map((name, i) => ({
            label: name,
            data: metrics.map(m => toPercent(models[name][m])),
            borderColor: PALETTE[i % PALETTE.length].replace('0.75', '1'),
            backgroundColor: PALETTE[i % PALETTE.length].replace('0.75', '0.20'),
            borderWidth: 2,
            pointRadius: 4,
        }));

        destroyIfExists(canvasId);

        new Chart(canvas, {
            type: 'radar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: v => v + '%', backdropColor: 'transparent' },
                        grid: { color: 'rgba(42, 42, 69, 0.6)' },
                        angleLines: { color: 'rgba(42, 42, 69, 0.6)' },
                        pointLabels: { color: '#a0a0b8', font: { size: 11 } }
                    }
                }
            }
        });
    }

    /* =========================================================
       11) Fetch helper
       ========================================================= */
    async function fetchJSON(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to fetch ${url}`);
        return res.json();
    }

    /* =========================================================
       12) Public API
       ========================================================= */
    window.PneumoCharts = {
        renderModelComparison,
        renderPreprocessingComparison,
        renderDatasetSplit,
        renderClassDistribution,
        renderConfusionMatrix,
        renderROCCurve,
        renderTrainingLoss,
        renderTrainingAccuracy,
        renderPredictionDistribution,
        renderModelRadar,
        fetchJSON,
        COLORS,
        PALETTE,
    };

})();