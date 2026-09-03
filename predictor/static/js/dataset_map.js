/**
 * Interactive Spatio-Temporal Crime Risk Map for India (2001–2013)
 * Uses Leaflet.js with state_risk_by_year.json, india_states.geojson,
 * and real district boundary outlines from india_districts.geojson
 */

(function () {
    'use strict';

    document.addEventListener('DOMContentLoaded', function () {
        const mapContainer = document.getElementById('indiaRiskMap');
        if (!mapContainer) return;

        const yearSelect = document.getElementById('mapYearSelect');
        const hoverTitle = document.getElementById('hoverStateName');
        const hoverDetails = document.getElementById('hoverStateDetails');
        const toggleDistrictCheckbox = document.getElementById('toggleDistrictMarkers');
        const resetZoomBtn = document.getElementById('btnResetMapZoom');

        // Color palette matching 3 risk tiers (Green, Amber, Red)
        const RISK_COLORS = {
            'Low': '#2E7D32',       // Green (<= 1500)
            'Moderate': '#F59E0B',  // Amber (1501 - 4000)
            'High': '#DC2626',      // Red (> 4000)
            'Default': '#94A3B8'    // Slate gray for unmapped
        };

        // State name normalization table
        function normalizeName(name) {
            if (!name) return '';
            const raw = String(name).trim().toUpperCase();
            const alias = {
                'ANDAMAN AND NICOBAR': 'ANDAMAN AND NICOBAR',
                'ANDAMAN & NICOBAR': 'ANDAMAN AND NICOBAR',
                'DADRA AND NAGAR HAVELI': 'DADRA AND NAGAR HAVELI',
                'D & N HAVELI': 'DADRA AND NAGAR HAVELI',
                'DAMAN AND DIU': 'DAMAN AND DIU',
                'DAMAN & DIU': 'DAMAN AND DIU',
                'DELHI': 'DELHI',
                'JAMMU AND KASHMIR': 'JAMMU AND KASHMIR',
                'JAMMU & KASHMIR': 'JAMMU AND KASHMIR',
                'ORISSA': 'ORISSA',
                'ODISHA': 'ORISSA',
                'UTTARANCHAL': 'UTTARANCHAL',
                'UTTARAKHAND': 'UTTARANCHAL',
                'TELANGANA': 'TELANGANA'
            };
            return alias[raw] || raw;
        }

        // Initialize Leaflet Map centered over India
        const map = L.map('indiaRiskMap', {
            center: [22.8, 80.5],
            zoom: 5,
            minZoom: 4,
            maxZoom: 11,
            zoomControl: true,
            attributionControl: false
        });

        // Add free, public OpenStreetMap base tile layer (no API key required)
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        let geojsonLayer = null;
        let districtBorderLayer = null;
        let riskData = null;
        let geojsonData = null;
        let districtGeojsonData = null;
        let districtStatsMap = null;

        function getStateData(stateName, selectedYear) {
            if (!riskData || !riskData.by_year || !riskData.by_year[selectedYear]) return null;
            const norm = normalizeName(stateName);
            return riskData.by_year[selectedYear][norm] || null;
        }

        function getDistrictData(stateName, districtName, selectedYear) {
            if (!districtStatsMap) return null;
            const s = String(stateName || '').trim().toUpperCase();
            const d = String(districtName || '').trim().toUpperCase();
            
            // Try direct key
            const key = `${s}___${d}`;
            if (districtStatsMap[key]) {
                const info = districtStatsMap[key];
                const count = info.yearly_crimes[selectedYear] !== undefined ? info.yearly_crimes[selectedYear] : info.latest_crime;
                const risk = info.yearly_risks[selectedYear] !== undefined ? info.yearly_risks[selectedYear] : info.latest_risk;
                return { district: info.district, state: info.state, count: count, risk: risk };
            }

            // Fuzzy match by district name alone
            for (const [k, val] of Object.entries(districtStatsMap)) {
                if (k.endsWith(`___${d}`) || d.includes(val.district.toUpperCase()) || val.district.toUpperCase().includes(d)) {
                    const count = val.yearly_crimes[selectedYear] !== undefined ? val.yearly_crimes[selectedYear] : val.latest_crime;
                    const risk = val.yearly_risks[selectedYear] !== undefined ? val.yearly_risks[selectedYear] : val.latest_risk;
                    return { district: val.district, state: val.state, count: count, risk: risk };
                }
            }

            return null;
        }

        function getStyle(feature) {
            const stateName = feature.properties.NAME_1 || feature.properties.name;
            const year = yearSelect ? yearSelect.value : '2013';
            const data = getStateData(stateName, year);

            const risk = data ? data.risk_level : 'Default';
            const fillColor = RISK_COLORS[risk] || RISK_COLORS['Default'];

            return {
                fillColor: fillColor,
                weight: 2,
                opacity: 1,
                color: '#ffffff',
                fillOpacity: 0.65
            };
        }

        // District boundary outline styling
        function getDistrictStyle() {
            return {
                fillColor: '#ffffff',
                fillOpacity: 0.02,
                color: '#334155',
                weight: 0.8,
                dashArray: '2, 3',
                opacity: 0.75
            };
        }

        function highlightDistrict(e) {
            const layer = e.target;
            const props = layer.feature.properties || {};
            const stateName = props.NAME_1 || 'Unknown State';
            const districtName = props.NAME_2 || 'District';
            const year = yearSelect ? yearSelect.value : '2013';

            const dData = getDistrictData(stateName, districtName, year);
            const count = dData ? dData.count : null;
            const risk = dData ? dData.risk : 'Moderate';
            const col = RISK_COLORS[risk] || '#F59E0B';

            layer.setStyle({
                weight: 2.5,
                color: '#0F172A',
                dashArray: '',
                fillColor: col,
                fillOpacity: 0.35
            });

            if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
                layer.bringToFront();
            }

            if (hoverTitle) hoverTitle.textContent = `${districtName} (${stateName})`;
            if (hoverDetails) {
                hoverDetails.innerHTML = `
                    <div class="mt-2" style="color: var(--color-heading);">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span style="color: var(--color-paragraph);">District Risk:</span>
                            <span class="badge" style="background-color:${col}; color:#fff; font-size:11px; padding:4px 8px;">${risk}</span>
                        </div>
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span style="color: var(--color-paragraph);">Year ${year} Crimes:</span>
                            <strong style="color: var(--color-heading); font-size:13px;">${count !== null ? count.toLocaleString() : 'N/A'}</strong>
                        </div>
                        <div class="d-flex justify-content-between align-items-center">
                            <span style="color: var(--color-paragraph);">Boundary Type:</span>
                            <span class="small" style="color: var(--color-paragraph);">Internal District Border</span>
                        </div>
                    </div>
                `;
            }
        }

        function resetDistrictHighlight(e) {
            if (districtBorderLayer) {
                districtBorderLayer.resetStyle(e.target);
            }
            if (hoverTitle) hoverTitle.textContent = 'Hover over a State / District';
            if (hoverDetails) hoverDetails.textContent = 'Select or hover over any state or internal district boundary outline to inspect crime counts and risk tiers.';
        }

        function onEachDistrictFeature(feature, layer) {
            const props = feature.properties || {};
            const distName = props.NAME_2 || 'District';
            const stateName = props.NAME_1 || 'State';

            layer.bindTooltip(`<strong>${distName}</strong><br><small style="color:#64748B;">${stateName}</small>`, {
                sticky: true,
                direction: 'top',
                className: 'shadow-sm'
            });

            layer.on({
                mouseover: highlightDistrict,
                mouseout: resetDistrictHighlight,
                click: function (e) {
                    map.fitBounds(e.target.getBounds(), { padding: [30, 30] });
                }
            });
        }

        function highlightFeature(e) {
            const layer = e.target;
            layer.setStyle({
                weight: 3,
                color: '#1E293B',
                fillOpacity: 0.85
            });

            const stateName = layer.feature.properties.NAME_1 || layer.feature.properties.name;
            const year = yearSelect ? yearSelect.value : '2013';
            const data = getStateData(stateName, year);

            if (hoverTitle) hoverTitle.textContent = stateName;
            if (hoverDetails) {
                if (data) {
                    const badgeClass = data.risk_level === 'Low' ? 'bg-success text-white'
                        : data.risk_level === 'Moderate' ? 'bg-warning text-dark'
                        : 'bg-danger text-white';

                    hoverDetails.innerHTML = `
                        <div class="mt-2" style="color: var(--color-heading);">
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span style="color: var(--color-paragraph);">State Risk Tier:</span>
                                <span class="badge ${badgeClass}" style="background-color:${RISK_COLORS[data.risk_level]}; font-size:11px; padding:4px 8px;">${data.risk_level}</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span style="color: var(--color-paragraph);">Avg/District:</span>
                                <strong style="color: var(--color-heading); font-size:13px;">${data.avg_crimes_per_district.toLocaleString()}</strong>
                            </div>
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span style="color: var(--color-paragraph);">Total Crimes:</span>
                                <strong style="color: var(--color-heading); font-size:13px;">${data.total_crimes.toLocaleString()}</strong>
                            </div>
                            <div class="d-flex justify-content-between align-items-center">
                                <span style="color: var(--color-paragraph);">Districts:</span>
                                <strong style="color: var(--color-heading); font-size:13px;">${data.district_count}</strong>
                            </div>
                        </div>
                    `;
                } else {
                    hoverDetails.innerHTML = '<span style="color: var(--color-paragraph);">No crime data recorded for this region.</span>';
                }
            }
        }

        function resetHighlight(e) {
            if (geojsonLayer) {
                geojsonLayer.resetStyle(e.target);
            }
            if (hoverTitle) hoverTitle.textContent = 'Hover over a State / District';
            if (hoverDetails) hoverDetails.textContent = 'Select or hover over any state or internal district boundary outline to inspect crime counts and risk tiers.';
        }

        function onEachFeature(feature, layer) {
            layer.on({
                mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: function (e) {
                    map.fitBounds(e.target.getBounds(), { padding: [25, 25] });
                }
            });
        }

        function updateMap() {
            if (geojsonLayer) {
                geojsonLayer.eachLayer(function (layer) {
                    layer.setStyle(getStyle(layer.feature));
                });
            }
        }

        // Fetch precomputed data, GeoJSON, district GeoJSON and stats map concurrently
        Promise.all([
            fetch('/static/data/state_risk_by_year.json').then(res => res.json()),
            fetch('/static/data/india_states.geojson').then(res => res.json()),
            fetch('/static/data/india_districts.geojson').then(res => res.json()),
            fetch('/static/data/district_stats_map.json').then(res => res.json())
        ])
            .then(function (results) {
                riskData = results[0];
                geojsonData = results[1];
                districtGeojsonData = results[2];
                districtStatsMap = results[3];

                // 1. Add State choropleth layer
                geojsonLayer = L.geoJSON(geojsonData, {
                    style: getStyle,
                    onEachFeature: onEachFeature
                }).addTo(map);

                // 2. Add District Boundary Outlines layer
                districtBorderLayer = L.geoJSON(districtGeojsonData, {
                    style: getDistrictStyle,
                    onEachFeature: onEachDistrictFeature
                }).addTo(map);

                // Auto-fit map viewport to India bounds
                try {
                    map.fitBounds(geojsonLayer.getBounds(), { padding: [10, 10] });
                } catch (e) {
                    console.log('[Map] Could not auto-fit bounds:', e);
                }

                if (yearSelect) {
                    yearSelect.addEventListener('change', updateMap);
                }

                if (toggleDistrictCheckbox) {
                    toggleDistrictCheckbox.addEventListener('change', function () {
                        if (this.checked) {
                            map.addLayer(districtBorderLayer);
                        } else {
                            map.removeLayer(districtBorderLayer);
                        }
                    });
                }

                if (resetZoomBtn) {
                    resetZoomBtn.addEventListener('click', function () {
                        if (geojsonLayer) {
                            map.fitBounds(geojsonLayer.getBounds(), { padding: [10, 10] });
                        }
                    });
                }
            })
            .catch(function (err) {
                console.error('[Map] Error loading map artifacts:', err);
                if (mapContainer) {
                    mapContainer.innerHTML = `
                        <div class="d-flex align-items-center justify-content-center h-100 text-muted">
                            <p><i class="bi bi-exclamation-triangle me-2"></i>Unable to load map data (${err.message}).</p>
                        </div>
                    `;
                }
            });
    });
})();
