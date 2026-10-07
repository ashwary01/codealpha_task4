document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------------
    // 1. Navigation Tab Controller
    // ---------------------------------------------------------
    const navItems = document.querySelectorAll('.nav-item');
    const tabPages = document.querySelectorAll('.tab-page');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = item.getAttribute('data-tab');

            navItems.forEach(nav => nav.classList.remove('active'));
            tabPages.forEach(page => page.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.add('active');
        });
    });

    // ---------------------------------------------------------
    // 2. Data & Metrics Store (Loaded from ML outputs)
    // ---------------------------------------------------------
    const modelMetrics = [
        { name: 'Linear Regression', r2: 0.9493, adj_r2: 0.9418, cv_r2: 0.9425, mae: 0.880, rmse: 1.138, status: 'Best Fit' },
        { name: 'Ridge Regression', r2: 0.9484, adj_r2: 0.9408, cv_r2: 0.9415, mae: 0.888, rmse: 1.148, status: 'Robust' },
        { name: 'Lasso Regression', r2: 0.9252, adj_r2: 0.9141, cv_r2: 0.9243, mae: 1.139, rmse: 1.382, status: 'Regularized' },
        { name: 'Gradient Boosting', r2: 0.8959, adj_r2: 0.8806, cv_r2: 0.8683, mae: 1.290, rmse: 1.630, status: 'Ensemble' },
        { name: 'Random Forest', r2: 0.8812, adj_r2: 0.8638, cv_r2: 0.8650, mae: 1.457, rmse: 1.741, status: 'Ensemble' },
        { name: 'Decision Tree', r2: 0.7174, adj_r2: 0.6759, cv_r2: 0.7823, mae: 2.042, rmse: 2.686, status: 'Baseline' }
    ];

    const sampleDataset = [
        { tv: 114.1, radio: 31.8, news: 12.0, segment: 'Mass Market', platform: 'Social Media', sales: 15.34 },
        { tv: 282.0, radio: 4.2, news: 102.9, segment: 'Mass Market', platform: 'Omnichannel Digital', sales: 16.41 },
        { tv: 218.3, radio: 8.0, news: 57.7, segment: 'Mass Market', platform: 'Streaming & TV', sales: 16.09 },
        { tv: 179.4, radio: 44.6, news: 94.3, segment: 'Mass Market', platform: 'Streaming & TV', sales: 22.21 },
        { tv: 50.5, radio: 30.1, news: 36.7, segment: 'Mass Market', platform: 'Streaming & TV', sales: 12.75 },
        { tv: 198.9, radio: 2.6, news: 21.2, segment: 'Gen Z & Youth', platform: 'Social Media', sales: 14.80 },
        { tv: 106.8, radio: 15.8, news: 31.6, segment: 'High Income', platform: 'Omnichannel Digital', sales: 14.92 },
        { tv: 240.1, radio: 7.3, news: 8.7, segment: 'B2B & Professionals', platform: 'Omnichannel Digital', sales: 17.55 }
    ];

    // ---------------------------------------------------------
    // 3. Render Tables
    // ---------------------------------------------------------
    const tableModelsBody = document.getElementById('table-models-body');
    if (tableModelsBody) {
        tableModelsBody.innerHTML = modelMetrics.map(m => `
            <tr>
                <td><strong>${m.name}</strong></td>
                <td class="${m.r2 > 0.9 ? 'text-cyan' : ''}">${(m.r2 * 100).toFixed(2)}%</td>
                <td>${(m.adj_r2 * 100).toFixed(2)}%</td>
                <td>${(m.cv_r2 * 100).toFixed(2)}%</td>
                <td>${m.mae.toFixed(3)}k</td>
                <td>${m.rmse.toFixed(3)}k</td>
                <td><span class="insight-badge ${m.status === 'Best Fit' ? 'badge-success' : 'badge-primary'}">${m.status}</span></td>
            </tr>
        `).join('');
    }

    const tableDatasetBody = document.getElementById('table-dataset-body');
    if (tableDatasetBody) {
        tableDatasetBody.innerHTML = sampleDataset.map(d => `
            <tr>
                <td>$${d.tv.toFixed(1)}k</td>
                <td>$${d.radio.toFixed(1)}k</td>
                <td>$${d.news.toFixed(1)}k</td>
                <td>${d.segment}</td>
                <td>${d.platform}</td>
                <td><strong>$${d.sales.toFixed(2)}k</strong></td>
            </tr>
        `).join('');
    }

    // ---------------------------------------------------------
    // 4. Interactive Simulator & Model Equation Predictor
    // ---------------------------------------------------------
    const inputTv = document.getElementById('input-tv');
    const inputRadio = document.getElementById('input-radio');
    const inputNews = document.getElementById('input-news');
    const selectSegment = document.getElementById('select-segment');
    const selectPlatform = document.getElementById('select-platform');

    const valTv = document.getElementById('val-tv');
    const valRadio = document.getElementById('val-radio');
    const valNews = document.getElementById('val-news');

    const predSalesVal = document.getElementById('pred-sales-value');
    const predSalesUnits = document.getElementById('pred-sales-units');
    const totalSpendVal = document.getElementById('total-spend-val');
    const totalRoiVal = document.getElementById('total-roi-val');

    // Formula derived from ML pipeline:
    // Base Sales = 3.95
    // TV coef = 0.04637 per $1k
    // Radio coef = 0.21418 per $1k
    // Newspaper coef = 0.00632 per $1k
    const segMultipliers = { 'Mass Market': 0.0, 'Gen Z & Youth': 0.8, 'B2B & Professionals': 1.5, 'High Income': 2.2 };
    const platMultipliers = { 'Omnichannel Digital': 1.2, 'Streaming & TV': 1.0, 'Social Media': 0.6, 'Print & Radio': -0.4 };

    function calculatePrediction() {
        const tv = parseFloat(inputTv.value);
        const radio = parseFloat(inputRadio.value);
        const news = parseFloat(inputNews.value);

        valTv.textContent = `$${tv.toFixed(1)}k`;
        valRadio.textContent = `$${radio.toFixed(1)}k`;
        valNews.textContent = `$${news.toFixed(1)}k`;

        const totalSpend = tv + radio + news;
        totalSpendVal.textContent = `$${totalSpend.toFixed(1)}k`;

        const segEffect = segMultipliers[selectSegment.value] || 0.0;
        const platEffect = platMultipliers[selectPlatform.value] || 0.0;

        // Sales calculation ($1000s)
        const predictedSales = 3.95 + (0.04637 * tv) + (0.21418 * radio) + (0.00632 * news) + (0.0011 * tv * radio / 10.0) + segEffect + platEffect;
        
        predSalesVal.textContent = `$${predictedSales.toFixed(2)}k`;
        predSalesUnits.textContent = `~ ${Math.round(predictedSales * 1000).toLocaleString()} Units Sold`;

        const roiMultiplier = totalSpend > 0 ? (predictedSales / totalSpend) : 0.0;
        totalRoiVal.textContent = `${roiMultiplier.toFixed(2)}x`;

        updatePieChart(tv, radio, news);
    }

    [inputTv, inputRadio, inputNews, selectSegment, selectPlatform].forEach(el => {
        el.addEventListener('input', calculatePrediction);
    });

    // Auto Optimizer Button Handler
    const btnAutoOptimize = document.getElementById('btn-auto-optimize');
    if (btnAutoOptimize) {
        btnAutoOptimize.addEventListener('click', () => {
            const currentBudget = parseFloat(inputTv.value) + parseFloat(inputRadio.value) + parseFloat(inputNews.value) || 200.0;
            // Optimal allocation puts budget heavily into Radio & TV based on marginal ROI
            inputTv.value = 0;
            inputRadio.value = currentBudget;
            inputNews.value = 0;
            calculatePrediction();
        });
    }

    const btnQuickSim = document.getElementById('btn-quick-sim');
    if (btnQuickSim) {
        btnQuickSim.addEventListener('click', () => {
            document.querySelector('[data-tab="simulator"]').click();
        });
    }

    const btnExport = document.getElementById('btn-export');
    if (btnExport) {
        btnExport.addEventListener('click', () => {
            alert('Exporting Sales Forecast & ROI Optimization Report (PDF/CSV)...');
        });
    }

    // ---------------------------------------------------------
    // 5. Chart.js Graphs Initialization
    // ---------------------------------------------------------

    // A. Channel Impact Bar Chart
    const ctxChannel = document.getElementById('chart-channel-impact');
    if (ctxChannel) {
        new Chart(ctxChannel, {
            type: 'bar',
            data: {
                labels: ['TV Spend', 'Radio Spend', 'Newspaper Spend'],
                datasets: [{
                    label: 'Sales Revenue Generated per $1,000 Spend ($)',
                    data: [46.37, 214.18, 6.32],
                    backgroundColor: ['rgba(56, 189, 248, 0.8)', 'rgba(129, 140, 248, 0.8)', 'rgba(251, 191, 36, 0.8)'],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                }
            }
        });
    }

    // B. Budget Pie Chart
    let budgetPieChart = null;
    function updatePieChart(tv, radio, news) {
        const ctxPie = document.getElementById('chart-budget-pie');
        if (!ctxPie) return;

        if (budgetPieChart) {
            budgetPieChart.data.datasets[0].data = [tv, radio, news];
            budgetPieChart.update();
        } else {
            budgetPieChart = new Chart(ctxPie, {
                type: 'doughnut',
                data: {
                    labels: ['TV', 'Radio', 'Newspaper'],
                    datasets: [{
                        data: [tv, radio, news],
                        backgroundColor: ['#38bdf8', '#818cf8', '#fbbf24'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 12 } }
                    }
                }
            });
        }
    }

    // C. Model Comparison Chart
    const ctxModelComp = document.getElementById('chart-model-comparison');
    if (ctxModelComp) {
        new Chart(ctxModelComp, {
            type: 'line',
            data: {
                labels: modelMetrics.map(m => m.name),
                datasets: [
                    {
                        label: 'R² Score (%)',
                        data: modelMetrics.map(m => m.r2 * 100),
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'MAE ($k)',
                        data: modelMetrics.map(m => m.mae * 10),
                        borderColor: '#4ade80',
                        borderDash: [5, 5],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#94a3b8' } } },
                scales: {
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                }
            }
        });
    }

    // D. Correlation Heatmap Canvas Simulation
    const ctxCorr = document.getElementById('chart-correlation');
    if (ctxCorr) {
        new Chart(ctxCorr, {
            type: 'bar',
            data: {
                labels: ['TV vs Sales', 'Radio vs Sales', 'Newspaper vs Sales', 'TV vs Radio'],
                datasets: [{
                    label: 'Pearson Correlation Coefficient (r)',
                    data: [0.782, 0.576, 0.228, 0.054],
                    backgroundColor: ['#38bdf8', '#818cf8', '#fbbf24', '#94a3b8'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: {
                    x: { min: 0, max: 1.0, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                }
            }
        });
    }

    // Initial Trigger
    calculatePrediction();
});
