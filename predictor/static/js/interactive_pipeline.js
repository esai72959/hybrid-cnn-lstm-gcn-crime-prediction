/**
 * interactive_pipeline.js
 * Powers the Interactive Architecture Pipeline (2-Way vs 3-Way toggle + Node Inspector)
 * and the Interactive Prediction Workflow Stepper across Problem & Objectives and Methodology pages.
 */

(function () {
    "use strict";

    // ============================================================
    // 1. PIPELINE NODE KNOWLEDGE BASE
    // ============================================================
    const NODE_DETAILS = {
        data_input: {
            title: "Historical Crime Records & Graph Geometry",
            badge: "Data Ingestion",
            badgeClass: "bg-primary text-white",
            desc: "Historical district-level crime data (2001–2014) covering 36 Indian States/UTs and 850+ districts, enriched with latitude/longitude geographic centroid coordinates.",
            inputShape: "Raw CSV Records (N=9,385, 34 raw columns)",
            operation: "Geographic Centroid Lookup & Spatial Indexing",
            outputShape: "Indexed Tabular Records + GPS Centroids (Lat, Lng)",
            hyperparams: "Format: UTF-8 CSV, Temporal Span: 13+ Years"
        },
        preprocessing: {
            title: "Data Cleaning, Normalization & Sliding Windows",
            badge: "Preprocessing",
            badgeClass: "bg-info text-dark",
            desc: "Outliers cleaned, missing records imputed, values scaled using StandardScaler, and multi-year sequences constructed via a 3-year sliding lag window.",
            inputShape: "Raw Crime Records (N=9,385)",
            operation: "StandardScaler: z = (x - μ) / σ + 3-Year Time Windowing",
            outputShape: "Scaled Feature Tensors: (Batch, 1, 33) & (Batch, 3, 33)",
            hyperparams: "Window Size: T = 3 years, Zero-Missing Tolerance"
        },
        features: {
            title: "33 Engineered Features & Spatial Adjacency Graph",
            badge: "Feature Engineering",
            badgeClass: "bg-success text-white",
            desc: "33 model-ready spatial-temporal features (crime rates, ratios, lag differences, regional indicators) coupled with an 850x850 k-NN geographic adjacency matrix.",
            inputShape: "Preprocessed Cleaned Tensors",
            operation: "Feature Scaling + Graph Adjacency: A_ij = exp(-d_ij² / 2σ²)",
            outputShape: "X_feat: (Batch, 33), A_tilde: (850, 850)",
            hyperparams: "Features: 33 Dimensions, k-NN Neighbors: k = 5"
        },
        branch_cnn: {
            title: "1D-CNN Branch (Spatial Pattern Extraction)",
            badge: "Spatial Branch",
            badgeClass: "bg-primary text-white",
            desc: "Extracts localized spatial correlations and inter-crime category interactions across the 33 statutory crime dimensions using 1D convolutional kernels.",
            inputShape: "Tensor (Batch, 1, 33)",
            operation: "Conv1D(filters=64, kernel=3, ReLU) -> MaxPool1D -> Dropout(0.2) -> Flatten",
            outputShape: "Dense Spatial Vector: (Batch, 64)",
            hyperparams: "Filters: 64, Kernel Size: 3, Stride: 1, Activation: ReLU"
        },
        branch_lstm: {
            title: "LSTM Branch (Temporal Recurrence Dynamics)",
            badge: "Temporal Branch",
            badgeClass: "bg-teal text-white",
            desc: "Captures multi-year sequential momentum, longitudinal trends, and temporal autocorrelation across 3-year sliding historical windows using recurrent memory cells.",
            inputShape: "Tensor (Batch, 3, 33)",
            operation: "LSTM(units=64, return_sequences=True) -> Dropout(0.2) -> LSTM(units=32) -> Hidden State",
            outputShape: "Dense Temporal Vector: (Batch, 32)",
            hyperparams: "Hidden Units: 64 -> 32, Recurrent Dropout: 0.2, Activation: Tanh"
        },
        branch_gcn: {
            title: "Spectral GCN Branch (Topological Graph Diffusion)",
            badge: "Topological Branch",
            badgeClass: "bg-warning text-dark",
            desc: "Models inter-district spatial proximity and cross-border crime spillover across 850 district nodes using normalized graph Laplacian convolutions.",
            inputShape: "Node Features (850, F) + Adjacency Matrix (850, 850)",
            operation: "Kipf-Welling GCN: Z = D̃^(-1/2) Ã D̃^(-1/2) X W",
            outputShape: "Dense Graph Embedding: (Batch, 32)",
            hyperparams: "Graph Layers: 1, Output Units: 32, Normalization: Kipf-Welling Symmetric"
        },
        fusion: {
            title: "Multi-Modal Dense Latent Feature Fusion",
            badge: "Fusion Layer",
            badgeClass: "bg-danger text-white",
            desc: "Concatenates spatial, temporal, and topological graph embeddings into a unified dense latent representation capturing all three dimensions simultaneously.",
            inputShape: "3-Way: [CNN: 64d, LSTM: 32d, GCN: 32d] | 2-Way: [CNN: 64d, LSTM: 32d]",
            operation: "Concatenate(axis=-1) -> Dense(64, ReLU) -> Dropout(0.2)",
            outputShape: "Unified Latent Embedding: (Batch, 128) [3-Way] / (Batch, 96) [2-Way]",
            hyperparams: "Dense Units: 64, Dropout Rate: 0.20, L2 Regularization: 1e-4"
        },
        dense_regressor: {
            title: "Multi-Layer Dense Regression Head",
            badge: "Regression Head",
            badgeClass: "bg-dark text-white",
            desc: "Maps the fused 128-dimensional latent vector through non-linear dense layers with dropout regularization to predict continuous real crime counts.",
            inputShape: "Dense Vector (Batch, 64)",
            operation: "Dense(32, ReLU) -> Dropout(0.1) -> Dense(1, Linear)",
            outputShape: "Scalar Prediction: (Batch, 1)",
            hyperparams: "Loss: MSE / Huber, Optimizer: Adam (lr=0.001)"
        },
        prediction_output: {
            title: "Crime Forecast & Tactical Risk Scoring",
            badge: "Output Stage",
            badgeClass: "bg-success text-white",
            desc: "Produces the unscaled real Total IPC crime count forecast and dynamically classifies tactical risk (Low, Moderate, High, Very High) with patrol advisory.",
            inputShape: "Scalar Prediction Tensor (Batch, 1)",
            operation: "Inverse Transform: y_pred * σ_y + μ_y -> Dynamic Percentile Risk Binning",
            outputShape: "Total IPC Count (e.g., 3,450) + Risk Tier + Tactical Advisory",
            hyperparams: "Target: Total IPC Crimes, Metric: R² = 98.10% (3-Way)"
        }
    };

    // ============================================================
    // 2. PREDICTION WORKFLOW STEPS KNOWLEDGE BASE
    // ============================================================
    const WORKFLOW_STEPS = [
        {
            num: "01",
            title: "State / UT Selection",
            tag: "Geographic Scope",
            desc: "The user selects an Indian State or Union Territory from 36 available administrative regions.",
            detail: "The platform filters district boundaries, regional baseline rates, and preloads state-specific spatial metadata.",
            inputData: "Selected State: e.g., 'TELANGANA' / 'MAHARASHTRA'",
            outputData: "Filtered District Roster (e.g., 33 districts in Telangana)",
            icon: "bi-map"
        },
        {
            num: "02",
            title: "District Selection & Centroid Query",
            tag: "Spatial Anchor",
            desc: "Target district is selected, triggering centroid GPS coordinate retrieval and k-NN spatial graph neighbor resolution.",
            detail: "Queries geographic coordinate database (e.g., Hyderabad at 17.3850° N, 78.4867° E) and resolves top 5 closest neighboring districts for graph adjacency.",
            inputData: "Selected District: e.g., 'HYDERABAD'",
            outputData: "Centroid: (17.3850° N, 78.4867° E), Neighbors: [Ranga Reddy (24km), Medchal (32km)...]",
            icon: "bi-geo-alt"
        },
        {
            num: "03",
            title: "Year / Forecast Horizon Selection",
            tag: "Temporal Horizon",
            desc: "User selects either a historical evaluation year (2001–2014) or a forward-looking future forecasting year (2015–2030+).",
            detail: "If historical, ground-truth records are attached for real-time validation; if future, multi-year auto-regressive projection is activated.",
            inputData: "Target Year: e.g., 2026",
            outputData: "Horizon Type: Future Multi-Year Projection (12 Years Forward)",
            icon: "bi-calendar-event"
        },
        {
            num: "04",
            title: "Feature Vector & Graph Ingestion",
            tag: "Tensor Assembly",
            desc: "System constructs the 33-dimensional engineered feature vector and 3-year sliding lag matrix for model ingestion.",
            detail: "Assembles normalized spatial features (1x33), 3-year sequential momentum tensor (3x33), and corresponding row from the 850x850 spatial adjacency matrix.",
            inputData: "33 Scaled Features + 3-Year Lag Time Series",
            outputData: "Tensors: X_spatial (1, 33), X_temporal (3, 33), A_graph (850, 850)",
            icon: "bi-sliders"
        },
        {
            num: "05",
            title: "Tri-Modal Hybrid Deep Learning Inference",
            tag: "Model Execution",
            desc: "Parallel forward execution through 1D-CNN, LSTM, and Spectral GCN branches simultaneously.",
            detail: "1D-CNN extracts 64-dim spatial embedding; LSTM computes 32-dim sequential temporal state; Spectral GCN propagates 32-dim topological spillover.",
            inputData: "Parallel Input Tensors",
            outputData: "Latent Embeddings: CNN (64d) + LSTM (32d) + GCN (32d)",
            icon: "bi-cpu"
        },
        {
            num: "06",
            title: "Latent Fusion & Continuous Crime Forecasting",
            tag: "Regression Head",
            desc: "Embeddings are concatenated into a 128-dimensional latent vector and passed through dense layers to predict unscaled crime count.",
            detail: "Fused 128-dim representation passes through Dense(64) -> Dense(32) -> Dense(1) to predict exact unscaled Total IPC crimes.",
            inputData: "128-dim Tri-Modal Latent Vector",
            outputData: "Predicted Total IPC Crimes: e.g., 14,820 incidents (±280 confidence)",
            icon: "bi-graph-up-arrow"
        },
        {
            num: "07",
            title: "Tactical Risk Scoring & Law Enforcement Advisory",
            tag: "Decision Support",
            desc: "Forecast is categorized into strategic risk tiers (Low, Moderate, High, Very High) with actionable policing recommendations.",
            detail: "Generates risk score (0–100), percentile threshold ranking, resource allocation advisory, and spatial spillover radar for adjacent districts.",
            inputData: "Predicted Crime Count + Historical Percentile Curve",
            outputData: "Risk Tier: 'High Risk' (Score: 88/100) -> Deploy targeted patrols in high-density corridors",
            icon: "bi-shield-check"
        }
    ];

    // ============================================================
    // 3. INITIALIZE INTERACTIVE PIPELINE DIAGRAM
    // ============================================================
    function initInteractivePipeline() {
        const pipelineWrap = document.getElementById("interactivePipelineWrap");
        if (!pipelineWrap) return;

        const toggleBtns = pipelineWrap.querySelectorAll("[data-pipeline-mode]");
        const gcnBranchNode = pipelineWrap.querySelector("#branchNodeGCN");
        const fusionDimBadge = pipelineWrap.querySelector("#fusionDimBadge");
        const pipelineMetricBadge = pipelineWrap.querySelector("#pipelineMetricBadge");
        const inspectorCard = document.getElementById("pipelineNodeInspector");
        const interactiveNodes = pipelineWrap.querySelectorAll(".pipeline-interactive-node");

        let currentMode = "3way";
        let activeNodeKey = "branch_gcn";

        function updateInspector(key) {
            const data = NODE_DETAILS[key];
            if (!data || !inspectorCard) return;

            activeNodeKey = key;
            interactiveNodes.forEach(node => {
                if (node.getAttribute("data-node-key") === key) {
                    node.classList.add("is-active-node");
                } else {
                    node.classList.remove("is-active-node");
                }
            });

            const titleEl = inspectorCard.querySelector("#inspectorTitle");
            const badgeEl = inspectorCard.querySelector("#inspectorBadge");
            const descEl = inspectorCard.querySelector("#inspectorDesc");
            const inputEl = inspectorCard.querySelector("#inspectorInput");
            const opEl = inspectorCard.querySelector("#inspectorOp");
            const outputEl = inspectorCard.querySelector("#inspectorOutput");
            const paramsEl = inspectorCard.querySelector("#inspectorParams");

            if (titleEl) titleEl.textContent = data.title;
            if (badgeEl) {
                badgeEl.textContent = data.badge;
                badgeEl.className = `badge ${data.badgeClass} rounded-pill px-3 py-1 font-monospace`;
            }
            if (descEl) descEl.textContent = data.desc;
            if (inputEl) inputEl.textContent = data.inputShape;
            if (opEl) opEl.textContent = data.operation;
            if (outputEl) outputEl.textContent = data.outputShape;
            if (paramsEl) paramsEl.textContent = data.hyperparams;

            inspectorCard.classList.add("inspector-updated");
            setTimeout(() => inspectorCard.classList.remove("inspector-updated"), 350);
        }

        function setMode(mode) {
            currentMode = mode;
            toggleBtns.forEach(btn => {
                if (btn.getAttribute("data-pipeline-mode") === mode) {
                    btn.classList.add("active", "btn-primary");
                    btn.classList.remove("btn-outline-primary");
                } else {
                    btn.classList.remove("active", "btn-primary");
                    btn.classList.add("btn-outline-primary");
                }
            });

            if (mode === "2way") {
                if (gcnBranchNode) {
                    gcnBranchNode.classList.add("node-dimmed");
                    gcnBranchNode.setAttribute("aria-hidden", "true");
                }
                if (fusionDimBadge) fusionDimBadge.textContent = "96-dim (CNN: 64d + LSTM: 32d)";
                if (pipelineMetricBadge) {
                    pipelineMetricBadge.textContent = "2-Way R² = 97.49% (CV: 96.47%)";
                    pipelineMetricBadge.className = "badge bg-info text-dark rounded-pill px-3 py-2";
                }
                if (activeNodeKey === "branch_gcn") {
                    updateInspector("branch_cnn");
                }
                pipelineWrap.classList.add("mode-2way");
                pipelineWrap.classList.remove("mode-3way");
            } else {
                if (gcnBranchNode) {
                    gcnBranchNode.classList.remove("node-dimmed");
                    gcnBranchNode.removeAttribute("aria-hidden");
                }
                if (fusionDimBadge) fusionDimBadge.textContent = "128-dim (CNN: 64d + LSTM: 32d + GCN: 32d)";
                if (pipelineMetricBadge) {
                    pipelineMetricBadge.textContent = "3-Way R² = 98.10% (CV: 96.10%) [Winner]";
                    pipelineMetricBadge.className = "badge bg-primary text-white rounded-pill px-3 py-2";
                }
                pipelineWrap.classList.add("mode-3way");
                pipelineWrap.classList.remove("mode-2way");
            }
        }

        toggleBtns.forEach(btn => {
            btn.addEventListener("click", function (e) {
                e.preventDefault();
                setMode(this.getAttribute("data-pipeline-mode"));
            });
        });

        interactiveNodes.forEach(node => {
            const key = node.getAttribute("data-node-key");
            node.addEventListener("mouseenter", () => {
                if (currentMode === "2way" && key === "branch_gcn") return;
                updateInspector(key);
            });
            node.addEventListener("click", (e) => {
                e.preventDefault();
                if (currentMode === "2way" && key === "branch_gcn") {
                    setMode("3way");
                }
                updateInspector(key);
            });
        });

        setMode("3way");
        updateInspector("branch_gcn");
    }

    // ============================================================
    // 4. INITIALIZE INTERACTIVE PREDICTION STEPPER
    // ============================================================
    function initInteractiveStepper() {
        const stepperWrap = document.getElementById("interactiveStepperWrap");
        if (!stepperWrap) return;

        const stepBtns = stepperWrap.querySelectorAll("[data-step-index]");
        const prevBtn = stepperWrap.querySelector("#stepperPrevBtn");
        const nextBtn = stepperWrap.querySelector("#stepperNextBtn");
        const autoPlayBtn = stepperWrap.querySelector("#stepperAutoPlayBtn");
        const progressBar = stepperWrap.querySelector("#stepperProgressBar");

        const displayNum = stepperWrap.querySelector("#stepDisplayNum");
        const displayTitle = stepperWrap.querySelector("#stepDisplayTitle");
        const displayTag = stepperWrap.querySelector("#stepDisplayTag");
        const displayDesc = stepperWrap.querySelector("#stepDisplayDesc");
        const displayDetail = stepperWrap.querySelector("#stepDisplayDetail");
        const displayInput = stepperWrap.querySelector("#stepDisplayInput");
        const displayOutput = stepperWrap.querySelector("#stepDisplayOutput");
        const displayIcon = stepperWrap.querySelector("#stepDisplayIcon");

        let currentStepIndex = 0;
        let autoPlayTimer = null;
        let isPlaying = false;

        function renderStep(index) {
            if (index < 0) index = 0;
            if (index >= WORKFLOW_STEPS.length) index = WORKFLOW_STEPS.length - 1;
            currentStepIndex = index;

            const data = WORKFLOW_STEPS[index];
            if (!data) return;

            stepBtns.forEach((btn, i) => {
                const stepNumEl = btn.querySelector(".stepper-badge");
                if (i === index) {
                    btn.classList.add("active-step");
                    btn.setAttribute("aria-selected", "true");
                    if (stepNumEl) {
                        stepNumEl.classList.remove("bg-light", "text-muted");
                        stepNumEl.classList.add("bg-primary", "text-white");
                    }
                } else if (i < index) {
                    btn.classList.remove("active-step");
                    btn.setAttribute("aria-selected", "false");
                    if (stepNumEl) {
                        stepNumEl.classList.remove("bg-light", "text-muted");
                        stepNumEl.classList.add("bg-success", "text-white");
                    }
                } else {
                    btn.classList.remove("active-step");
                    btn.setAttribute("aria-selected", "false");
                    if (stepNumEl) {
                        stepNumEl.classList.remove("bg-primary", "bg-success", "text-white");
                        stepNumEl.classList.add("bg-light", "text-muted");
                    }
                }
            });

            if (progressBar) {
                const pct = Math.round(((index + 1) / WORKFLOW_STEPS.length) * 100);
                progressBar.style.width = `${pct}%`;
                progressBar.setAttribute("aria-valuenow", pct);
            }

            if (displayNum) displayNum.textContent = data.num;
            if (displayTitle) displayTitle.textContent = data.title;
            if (displayTag) displayTag.textContent = data.tag;
            if (displayDesc) displayDesc.textContent = data.desc;
            if (displayDetail) displayDetail.textContent = data.detail;
            if (displayInput) displayInput.textContent = data.inputData;
            if (displayOutput) displayOutput.textContent = data.outputData;
            if (displayIcon) displayIcon.className = `bi ${data.icon} fs-1 text-primary`;

            if (prevBtn) prevBtn.disabled = index === 0;
            if (nextBtn) {
                if (index === WORKFLOW_STEPS.length - 1) {
                    nextBtn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i> Restart';
                } else {
                    nextBtn.innerHTML = 'Next Step <i class="bi bi-arrow-right ms-1"></i>';
                }
            }
        }

        function toggleAutoPlay() {
            if (isPlaying) {
                clearInterval(autoPlayTimer);
                autoPlayTimer = null;
                isPlaying = false;
                if (autoPlayBtn) {
                    autoPlayBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i> Auto-Play';
                    autoPlayBtn.classList.remove("btn-danger");
                    autoPlayBtn.classList.add("btn-outline-primary");
                }
            } else {
                isPlaying = true;
                if (autoPlayBtn) {
                    autoPlayBtn.innerHTML = '<i class="bi bi-pause-fill me-1"></i> Pause';
                    autoPlayBtn.classList.remove("btn-outline-primary");
                    autoPlayBtn.classList.add("btn-danger");
                }
                autoPlayTimer = setInterval(() => {
                    let next = (currentStepIndex + 1) % WORKFLOW_STEPS.length;
                    renderStep(next);
                }, 3500);
            }
        }

        stepBtns.forEach((btn, idx) => {
            btn.addEventListener("click", () => {
                if (isPlaying) toggleAutoPlay();
                renderStep(idx);
            });
        });

        if (prevBtn) {
            prevBtn.addEventListener("click", () => {
                if (isPlaying) toggleAutoPlay();
                renderStep(currentStepIndex - 1);
            });
        }

        if (nextBtn) {
            nextBtn.addEventListener("click", () => {
                if (isPlaying) toggleAutoPlay();
                if (currentStepIndex === WORKFLOW_STEPS.length - 1) {
                    renderStep(0);
                } else {
                    renderStep(currentStepIndex + 1);
                }
            });
        }

        if (autoPlayBtn) {
            autoPlayBtn.addEventListener("click", toggleAutoPlay);
        }

        renderStep(0);
    }

    // ============================================================
    // 5. DOM READY BOOTSTRAP
    // ============================================================
    document.addEventListener("DOMContentLoaded", function () {
        initInteractivePipeline();
        initInteractiveStepper();
    });

})();
