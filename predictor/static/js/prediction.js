/**
 * Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction
 * Prediction Page JavaScript Engine (prediction.js)
 *
 * Handles: states/districts dropdowns, form validation, CSRF security,
 * /api/predict/ execution, UI feedback, count-up + gauge animation.
 */

'use strict';

window.CPS = window.CPS || {};

// Guard against the script (or DOMContentLoaded) firing twice.
if (window.CPS.__predictionInitialized) {
    console.log('[Prediction] Already initialized — skipping duplicate init.');
} else {
    window.CPS.__predictionInitialized = true;

    CPS.Prediction = (() => {
        const elements = {};
        let isSubmitting = false;

        // Locked brand palette only (style.css :root) — this site never uses
        // green/amber/red alert colors, so risk levels stay tonal like the
        // rest of the design system. These are CSS custom properties, not
        // hex codes, so they re-tint automatically in dark mode.
        const RISK_TOKENS = {
            low: 'var(--color-primary)',
            medium: 'var(--color-secondary)',
            high: 'var(--color-hover)',
            default: 'var(--color-primary)'
        };

        /* ---------------------------------------------------------------- */
        /* DOM CACHING                                                       */
        /* ---------------------------------------------------------------- */
        const cacheElements = () => {
            elements.form = document.getElementById('predictionInputForm');
            elements.stateSelect = document.getElementById('state');
            elements.districtSelect = document.getElementById('district');
            elements.yearInput = document.getElementById('year');
            elements.modelTypeSelect = document.getElementById('modelType');
            elements.submitBtn = document.getElementById('runPredictionBtn');

            elements.resultSection = document.getElementById('resultSection');
            elements.predictionValue = document.getElementById('predictionValue');
            elements.riskCard = document.getElementById('riskCard');
            elements.riskLevel = document.getElementById('riskLevel');
            elements.confidenceValue = document.getElementById('confidenceValue');
            elements.confidenceBarFill = document.getElementById('confidenceBarFill');
            elements.recommendation = document.getElementById('recommendation');

            elements.resultState = document.getElementById('resultState');
            elements.resultDistrict = document.getElementById('resultDistrict');
            elements.resultYear = document.getElementById('resultYear');

            elements.riskGaugeFill = document.getElementById('riskGaugeFill');
            elements.riskGaugeLabel = document.getElementById('riskGaugeLabel');

            elements.statusMessage = document.getElementById('predictionStatusMessage');

            // Cards that get the reveal-on-scroll treatment (site-wide class
            // already defined in style.css — we just toggle .is-revealed).
            elements.revealTargets = elements.resultSection
                ? elements.resultSection.querySelectorAll('.reveal-on-scroll')
                : [];

            console.log('[Prediction] Elements cached:', {
                form: !!elements.form,
                stateSelect: !!elements.stateSelect,
                districtSelect: !!elements.districtSelect,
                yearInput: !!elements.yearInput,
                submitBtn: !!elements.submitBtn,
                resultSection: !!elements.resultSection
            });
        };

        /* ---------------------------------------------------------------- */
        /* CSRF                                                               */
        /* ---------------------------------------------------------------- */
        const getCsrfToken = () => {
            // Prefer the hidden {% csrf_token %} input Django renders inside the
            // form — this still works even if the csrftoken cookie is HttpOnly.
            const inputToken = document.querySelector('input[name="csrfmiddlewaretoken"]');
            if (inputToken && inputToken.value) return inputToken.value;

            let cookieValue = null;
            if (document.cookie) {
                const cookies = document.cookie.split(';');
                for (let i = 0; i < cookies.length; i++) {
                    const cookie = cookies[i].trim();
                    if (cookie.substring(0, 10) === 'csrftoken=') {
                        cookieValue = decodeURIComponent(cookie.substring(10));
                        break;
                    }
                }
            }
            return cookieValue;
        };

        /* ---------------------------------------------------------------- */
        /* STATUS MESSAGES                                                   */
        /* ---------------------------------------------------------------- */
        const showMessage = (message, type = 'danger') => {
            if (!elements.statusMessage) {
                if (type === 'danger' && message) alert(message);
                return;
            }
            const variant = type === 'warning' ? 'pred-status-message--warning'
                : type === 'danger' ? 'pred-status-message--danger'
                : '';
            elements.statusMessage.className = `pred-status-message ${variant}`.trim();
            elements.statusMessage.textContent = message;
        };

        const clearMessage = () => {
            if (!elements.statusMessage) return;
            elements.statusMessage.textContent = '';
            elements.statusMessage.className = 'pred-status-message is-hidden';
        };

        /* ---------------------------------------------------------------- */
        /* SELECT HELPERS                                                     */
        /* ---------------------------------------------------------------- */
        const populateSelect = (selectEl, options, defaultText, enable = true) => {
            if (!selectEl) return;
            selectEl.innerHTML = `<option value="" selected disabled>${defaultText}</option>`;
            options.forEach((item) => {
                // Support both plain strings and {name}/{id} style objects.
                const value = typeof item === 'string' ? item : (item.name || item.value || item.id || String(item));
                const opt = document.createElement('option');
                opt.value = value;
                opt.textContent = value;
                selectEl.appendChild(opt);
            });
            selectEl.disabled = !enable;
        };

        /**
         * Normalize an API response that may be:
         *  - { status: "success", states: [...] }  (or districts)
         *  - a bare array [...]
         *  - null / unexpected shape
         */
        const extractList = (data, key) => {
            if (Array.isArray(data)) return data;
            if (data && Array.isArray(data[key])) return data[key];
            return null;
        };

        const looksLikeAuthRedirect = (response) => {
            return response.redirected && /login/i.test(response.url);
        };

        /* ---------------------------------------------------------------- */
        /* LOAD STATES                                                        */
        /* ---------------------------------------------------------------- */
        const loadStates = async () => {
            if (!elements.stateSelect) return;
            console.log('[Prediction] Loading states...');

            elements.stateSelect.disabled = true;
            elements.stateSelect.innerHTML = '<option value="" selected disabled>Loading states...</option>';
            if (elements.districtSelect) {
                elements.districtSelect.disabled = true;
                elements.districtSelect.innerHTML = '<option value="" selected disabled>Select State First</option>';
            }

            try {
                const response = await fetch('/api/states/', {
                    method: 'GET',
                    headers: { 'Accept': 'application/json' },
                    credentials: 'same-origin',
                    cache: 'no-store'
                });

                if (looksLikeAuthRedirect(response)) {
                    throw new Error('Session expired — please log in again.');
                }

                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }

                let data;
                try {
                    data = await response.json();
                } catch (parseErr) {
                    throw new Error('Server did not return valid JSON.');
                }

                console.log('[Prediction] States API response:', data);

                const states = extractList(data, 'states');

                if (!states || states.length === 0) {
                    throw new Error('No states data received.');
                }

                console.log('[Prediction] States loaded:', states.length);
                populateSelect(elements.stateSelect, states, 'Select State');
            } catch (error) {
                console.error('[Prediction] Failed to load states:', error);
                elements.stateSelect.innerHTML = '<option value="" selected disabled>Error loading states</option>';
                elements.stateSelect.disabled = false;
                showMessage(`Unable to load states list: ${error.message}`, 'danger');
            }
        };

        /* ---------------------------------------------------------------- */
        /* LOAD DISTRICTS                                                     */
        /* ---------------------------------------------------------------- */
        const loadDistricts = async (state) => {
            if (!elements.districtSelect) return;
            console.log('[Prediction] Loading districts for state:', state);

            elements.districtSelect.disabled = true;
            elements.districtSelect.innerHTML = '<option value="" selected disabled>Loading districts...</option>';

            try {
                const url = `/api/districts/?state=${encodeURIComponent(state)}`;
                const response = await fetch(url, {
                    method: 'GET',
                    headers: { 'Accept': 'application/json' },
                    credentials: 'same-origin',
                    cache: 'no-store'
                });

                if (looksLikeAuthRedirect(response)) {
                    throw new Error('Session expired — please log in again.');
                }

                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }

                let data;
                try {
                    data = await response.json();
                } catch (parseErr) {
                    throw new Error('Server did not return valid JSON.');
                }

                console.log('[Prediction] Districts API response:', data);

                const districts = extractList(data, 'districts');

                if (!districts || districts.length === 0) {
                    elements.districtSelect.innerHTML = '<option value="" selected disabled>No districts available</option>';
                    elements.districtSelect.disabled = true;
                    return;
                }

                console.log('[Prediction] Districts loaded:', districts.length);
                populateSelect(elements.districtSelect, districts, 'Select District');
            } catch (error) {
                console.error('[Prediction] Failed to load districts:', error);
                elements.districtSelect.innerHTML = '<option value="" selected disabled>Error loading districts</option>';
                elements.districtSelect.disabled = false;
                showMessage(`Unable to load districts: ${error.message}`, 'danger');
            }
        };

        /* ---------------------------------------------------------------- */
        /* VALIDATION                                                         */
        /* ---------------------------------------------------------------- */
        const validateForm = () => {
            clearMessage();

            const state = elements.stateSelect ? elements.stateSelect.value.trim() : '';
            const district = elements.districtSelect ? elements.districtSelect.value.trim() : '';
            const yearRaw = elements.yearInput ? elements.yearInput.value.trim() : '';
            const year = parseInt(yearRaw, 10);

            if (!state) {
                showMessage('Please select a State / UT.', 'warning');
                if (elements.stateSelect) elements.stateSelect.focus();
                return null;
            }
            if (!district) {
                showMessage('Please select a District.', 'warning');
                if (elements.districtSelect) elements.districtSelect.focus();
                return null;
            }
            if (!yearRaw || isNaN(year) || year < 2025 || year > 2030) {
                showMessage('Please select a valid forecast year (2025 - 2030).', 'warning');
                if (elements.yearInput) elements.yearInput.focus();
                return null;
            }

            const modelType = elements.modelTypeSelect ? elements.modelTypeSelect.value.trim() : 'hybrid_gcn';

            return { state, district, year: String(year), model_type: modelType };
        };

        /* ---------------------------------------------------------------- */
        /* LOADING STATE                                                      */
        /* ---------------------------------------------------------------- */
        const setLoadingState = (isLoading) => {
            if (!elements.submitBtn) return;

            const label = elements.submitBtn.querySelector('.btn-predict-label');

            if (isLoading) {
                elements.submitBtn.disabled = true;
                elements.submitBtn.classList.add('is-loading');
                if (label) label.textContent = 'Running Prediction...';
            } else {
                elements.submitBtn.disabled = false;
                elements.submitBtn.classList.remove('is-loading');
                if (label) label.textContent = 'Run Prediction';
            }
        };

        /* ---------------------------------------------------------------- */
        /* ANIMATIONS                                                         */
        /* ---------------------------------------------------------------- */
        const animateCount = (el, target) => {
            const finalValue = Number(target) || 0;
            const duration = 900;
            const startTime = performance.now();

            const step = (now) => {
                const progress = Math.min((now - startTime) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
                const current = Math.round(finalValue * eased);
                el.textContent = current.toLocaleString();
                if (progress < 1) {
                    requestAnimationFrame(step);
                } else {
                    el.textContent = finalValue.toLocaleString();
                }
            };
            requestAnimationFrame(step);
        };

        const updateGauge = (riskScore, riskLevelText) => {
            const score = Math.max(0, Math.min(100, parseFloat(riskScore) || 0));
            const normalizedLevel = (riskLevelText || '').toLowerCase();

            let colorToken = RISK_TOKENS.default;
            if (normalizedLevel.includes('low')) colorToken = RISK_TOKENS.low;
            else if (normalizedLevel.includes('med')) colorToken = RISK_TOKENS.medium;
            else if (normalizedLevel.includes('high')) colorToken = RISK_TOKENS.high;

            if (elements.riskGaugeLabel) {
                elements.riskGaugeLabel.textContent = `${Math.round(score)}%`;
                elements.riskGaugeLabel.style.fill = colorToken;
            }

            if (elements.riskGaugeFill) {
                const radius = elements.riskGaugeFill.r.baseVal.value || 90;
                const circumference = 2 * Math.PI * radius;
                const offset = circumference - (score / 100) * circumference;

                elements.riskGaugeFill.style.stroke = colorToken;
                elements.riskGaugeFill.setAttribute('stroke-dasharray', `${circumference}`);
                // Force a reflow so the transition (defined in CSS) actually animates
                // from the current offset to the new one, instead of jumping.
                // eslint-disable-next-line no-unused-expressions
                elements.riskGaugeFill.getBoundingClientRect();
                elements.riskGaugeFill.style.strokeDashoffset = offset.toString();
            }
        };

        const revealResultCards = () => {
            if (!elements.revealTargets) return;
            elements.revealTargets.forEach((card, index) => {
                card.classList.remove('is-revealed');
                card.style.transitionDelay = `${index * 80}ms`;
                // Restart the transition even if it already ran once.
                void card.offsetWidth;
                card.classList.add('is-revealed');
            });
        };

        let districtNeighborsCache = null;

        const loadNeighborsData = async () => {
            if (districtNeighborsCache) return districtNeighborsCache;
            try {
                const res = await fetch('/static/data/district_neighbors.json?t=' + Date.now(), { cache: 'no-store' });
                if (res.ok) {
                    districtNeighborsCache = await res.json();
                }
            } catch (err) {
                console.error('[Prediction] Error loading district neighbors:', err);
            }
            return districtNeighborsCache;
        };

        const renderSpatialNeighborGraph = async (state, district, directNeighbors) => {
            const card = document.getElementById('spatialNeighborCard');
            const svg = document.getElementById('neighborGraphSvg');
            const list = document.getElementById('neighborNodesList');
            if (!card || !svg || !list) return;

            const targetName = String(district || '').trim();
            const targetState = String(state || '').trim();

            let neighbors = null;
            if (Array.isArray(directNeighbors) && directNeighbors.length > 0 && directNeighbors.some(n => Number(n.distance_km) > 0)) {
                neighbors = directNeighbors.slice(0, 5);
            } else {
                const neighborsData = await loadNeighborsData();
                if (neighborsData) {
                    const stNorm = targetState.toUpperCase();
                    const dtNorm = targetName.toUpperCase();
                    
                    const STATE_ALIASES = {
                        "UTTARANCHAL": "UTTARAKHAND",
                        "UTTARAKHAND": "UTTARAKHAND",
                        "ORISSA": "ODISHA",
                        "ODISHA": "ODISHA",
                        "DELHI": "DELHI UT",
                        "DELHI UT": "DELHI UT",
                        "PONDICHERRY": "PUDUCHERRY",
                        "PUDUCHERRY": "PUDUCHERRY",
                        "A & N ISLANDS": "A & N ISLANDS",
                        "ANDAMAN & NICOBAR": "A & N ISLANDS",
                        "ANDAMAN AND NICOBAR": "A & N ISLANDS",
                        "D & N HAVELI": "D & N HAVELI",
                        "DADRA & NAGAR HAVELI": "D & N HAVELI",
                        "DADRA AND NAGAR HAVELI": "D & N HAVELI",
                        "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
                        "JAMMU & KASHMIR": "JAMMU & KASHMIR",
                        "TELANGANA": "ANDHRA PRADESH"
                    };
                    
                    let info = neighborsData[`${stNorm}___${dtNorm}`];
                    if (!info) {
                        const aliasSt = STATE_ALIASES[stNorm] || stNorm;
                        info = neighborsData[`${aliasSt}___${dtNorm}`];
                    }
                    if (!info) {
                        for (const [k, val] of Object.entries(neighborsData)) {
                            if (k.endsWith(`___${dtNorm}`)) {
                                info = val;
                                break;
                            }
                        }
                    }
                    if (info && info.neighbors) {
                        neighbors = info.neighbors.slice(0, 5);
                    }
                }
            }

            if (!neighbors || neighbors.length === 0) {
                card.style.display = 'none';
                return;
            }

            card.style.display = 'block';

            const RISK_COLORS = {
                'Low': '#2E7D32',
                'Moderate': '#F59E0B',
                'High': '#DC2626'
            };

            const cx = 250, cy = 140;
            const radiusX = 170, radiusY = 85;

            let svgContent = `
                <defs>
                    <radialGradient id="neighborCenterGlow" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stop-color="var(--radar-ring-pulse, #3B82F6)" stop-opacity="0.6"/>
                        <stop offset="100%" stop-color="var(--radar-ring-pulse, #3B82F6)" stop-opacity="0"/>
                    </radialGradient>
                </defs>
            `;

            // Draw connecting lines and distance pills
            neighbors.forEach((n, i) => {
                const angle = (i * (2 * Math.PI / neighbors.length)) - (Math.PI / 2);
                const nx = cx + radiusX * Math.cos(angle);
                const ny = cy + radiusY * Math.sin(angle);

                const midX = (cx + nx) / 2;
                const midY = (cy + ny) / 2;

                svgContent += `
                    <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" 
                          stroke="var(--radar-link-line, #8CA3C6)" stroke-width="1.8" stroke-dasharray="4 4"/>
                    <rect x="${midX - 32}" y="${midY - 11}" width="64" height="22" rx="6" 
                          fill="var(--color-card, #1F2C3D)" stroke="var(--color-border, #8CA3C6)" stroke-width="1.2"/>
                    <text x="${midX}" y="${midY + 4}" text-anchor="middle" font-size="10.5" font-weight="700" 
                          fill="var(--color-primary, #8CA3C6)">${n.distance_km} km</text>
                `;
            });

            // Target Node in Center
            const displayName = targetName.length > 11 ? targetName.slice(0, 10) + '..' : targetName;
            svgContent += `
                <circle cx="${cx}" cy="${cy}" r="46" fill="url(#neighborCenterGlow)"/>
                <circle cx="${cx}" cy="${cy}" r="27" fill="var(--radar-center-bg, #24344D)" stroke="var(--color-primary, #8CA3C6)" stroke-width="3"/>
                <text x="${cx}" y="${cy + 4}" text-anchor="middle" font-size="10.5" font-weight="700" fill="#FFFFFF">${displayName}</text>
                <text x="${cx}" y="${cy + 40}" text-anchor="middle" font-size="9.5" font-weight="800" fill="var(--color-primary, #8CA3C6)" letter-spacing="0.06em">TARGET NODE</text>
            `;

            // Neighbor Nodes
            neighbors.forEach((n, i) => {
                const angle = (i * (2 * Math.PI / neighbors.length)) - (Math.PI / 2);
                const nx = cx + radiusX * Math.cos(angle);
                const ny = cy + radiusY * Math.sin(angle);
                const col = RISK_COLORS[n.risk_level] || '#64748B';

                svgContent += `
                    <g style="cursor: pointer;">
                        <circle cx="${nx}" cy="${ny}" r="18" fill="var(--color-card, #1F2C3D)" stroke="${col}" stroke-width="3"/>
                        <circle cx="${nx}" cy="${ny}" r="7" fill="${col}"/>
                        <text x="${nx}" y="${ny - 22}" text-anchor="middle" font-size="11" font-weight="700" fill="var(--color-heading, #F4F6F9)">${n.district}</text>
                    </g>
                `;
            });

            svg.innerHTML = svgContent;

            list.innerHTML = neighbors.map(n => `
                <div class="p-2 px-3 rounded d-flex justify-content-between align-items-center" style="background: var(--color-bg); border: 1px solid var(--color-border); font-size: 12px; transition: background var(--transition-fast);">
                    <div>
                        <strong style="color: var(--color-heading); display: block; font-size: 13px; font-weight: 700;">${n.district}</strong>
                        <span style="color: var(--color-paragraph); font-size: 11.5px;">${n.state} &bull; <strong style="color: var(--color-primary); font-weight: 700;">${n.distance_km} km away</strong></span>
                    </div>
                    <span class="badge" style="background-color: ${RISK_COLORS[n.risk_level] || '#64748B'}; color: #fff; font-size: 11px; font-weight: 700; padding: 5px 10px; border-radius: 12px;">
                        ${n.risk_level} (${Number(n.crime_2013 || 0).toLocaleString()})
                    </span>
                </div>
            `).join('');
        };

        /* ---------------------------------------------------------------- */
        /* RENDER RESULTS                                                     */
        /* ---------------------------------------------------------------- */
        const pickField = (data, keys, fallback) => {
            for (const key of keys) {
                if (data[key] !== undefined && data[key] !== null && data[key] !== '') {
                    return data[key];
                }
            }
            return fallback;
        };

        const renderResults = (data, requestInfo) => {
            const predictedCount = pickField(data, ['predicted_count', 'prediction', 'predicted_crime_count'], 0);
            const riskLevel = pickField(data, ['risk_level', 'riskLevel'], 'N/A');
            const confidence = pickField(data, ['confidence', 'confidence_score'], null);
            const recommendation = pickField(data, ['recommendation', 'advice'], 'No recommendation provided.');
            const riskScore = pickField(data, ['risk_score', 'riskScore'], confidence || 0);

            if (elements.predictionValue) {
                animateCount(elements.predictionValue, predictedCount);
            }

            if (elements.resultState && requestInfo) elements.resultState.textContent = requestInfo.state;
            if (elements.resultDistrict && requestInfo) elements.resultDistrict.textContent = requestInfo.district;
            if (elements.resultYear && requestInfo) elements.resultYear.textContent = requestInfo.year;

            if (elements.riskLevel) {
                elements.riskLevel.textContent = riskLevel;
            }

            if (elements.riskCard) {
                const level = String(riskLevel).toLowerCase();
                elements.riskCard.classList.remove('risk-card--low', 'risk-card--medium', 'risk-card--high', 'risk-card--very-high');
                if (level.includes('very')) elements.riskCard.classList.add('risk-card--very-high');
                else if (level.includes('high')) elements.riskCard.classList.add('risk-card--high');
                else if (level.includes('med')) elements.riskCard.classList.add('risk-card--medium');
                else elements.riskCard.classList.add('risk-card--low');
            }

            const conf = parseFloat(confidence);
            if (elements.confidenceValue) {
                elements.confidenceValue.textContent = !isNaN(conf) ? `${conf.toFixed(2)}%` : 'N/A';
            }
            if (elements.confidenceBarFill) {
                elements.confidenceBarFill.style.width = !isNaN(conf) ? `${Math.max(0, Math.min(100, conf))}%` : '0%';
            }

            if (elements.recommendation) {
                elements.recommendation.textContent = recommendation;
            }

            updateGauge(riskScore, riskLevel);

            if (requestInfo && requestInfo.state && requestInfo.district) {
                renderSpatialNeighborGraph(requestInfo.state, requestInfo.district, data.neighbors);
            }

            if (elements.resultSection) {
                elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }

            // Slight delay so the scroll has started before cards animate in.
            setTimeout(revealResultCards, 150);
        };

        /* ---------------------------------------------------------------- */
        /* FORM SUBMIT                                                        */
        /* ---------------------------------------------------------------- */
        const handleFormSubmit = async (e) => {
            e.preventDefault();

            if (isSubmitting) {
                console.log('[Prediction] Submission already in progress — ignoring click.');
                return;
            }

            const formData = validateForm();
            if (!formData) return;

            console.log('[Prediction] Running prediction...');
            console.log('[Prediction] Prediction request:', formData);

            isSubmitting = true;
            setLoadingState(true);

            try {
                const csrfToken = getCsrfToken();
                const headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                };
                if (csrfToken) headers['X-CSRFToken'] = csrfToken;

                const response = await fetch('/api/predict/', {
                    method: 'POST',
                    headers: headers,
                    credentials: 'same-origin',
                    body: JSON.stringify(formData)
                });

                if (looksLikeAuthRedirect(response)) {
                    throw new Error('Session expired — please log in again.');
                }

                let result;
                try {
                    result = await response.json();
                } catch (parseErr) {
                    throw new Error('Server did not return valid JSON.');
                }

                console.log('[Prediction] Prediction response:', result);

                if (!response.ok) {
                    throw new Error(result.message || result.error || `Server responded with status ${response.status}`);
                }

                if (result.status && result.status !== 'success') {
                    throw new Error(result.message || 'Prediction execution failed.');
                }

                clearMessage();
                renderResults(result, formData);
            } catch (error) {
                console.error('[Prediction] Prediction failed:', error);
                showMessage(`Prediction Error: ${error.message}`, 'danger');
            } finally {
                setLoadingState(false);
                isSubmitting = false;
            }
        };

        /* ---------------------------------------------------------------- */
        /* EVENTS                                                             */
        /* ---------------------------------------------------------------- */
        const bindEvents = () => {
            if (elements.form) {
                elements.form.addEventListener('submit', handleFormSubmit);
            } else if (elements.submitBtn) {
                elements.submitBtn.addEventListener('click', handleFormSubmit);
            }

            if (elements.stateSelect) {
                elements.stateSelect.addEventListener('change', (e) => {
                    const selectedState = e.target.value;
                    if (elements.districtSelect) {
                        elements.districtSelect.innerHTML = '<option value="" selected disabled>Select State First</option>';
                    }
                    if (selectedState) {
                        loadDistricts(selectedState);
                    }
                });
            }
        };

        /* ---------------------------------------------------------------- */
        /* INIT                                                               */
        /* ---------------------------------------------------------------- */
        const init = () => {
            console.log('[Prediction] Initializing');
            cacheElements();
            bindEvents();
            loadStates();
        };

        return { init };
    })();

    const start = () => CPS.Prediction.init();

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        // DOM is already ready (script loaded late / cached) — init immediately.
        start();
    }
}