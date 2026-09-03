/* ============================================================
   DATASET PAGE JAVASCRIPT
   Handles:
   1. State → District dependent filtering
   2. Search/filter form behaviour
   3. Loading state feedback
   4. Reset behaviour
============================================================ */

document.addEventListener("DOMContentLoaded", function () {

    const stateFilter = document.getElementById("stateFilter");
    const districtFilter = document.getElementById("districtFilter");
    const datasetForm = document.querySelector(
        'form[action*="dataset"]'
    );

    /* ========================================================
       STATE → DISTRICT DEPENDENT FILTER
    ======================================================== */

    if (stateFilter && districtFilter) {

        stateFilter.addEventListener("change", function () {

            const selectedState = this.value.trim();

            // Reset district dropdown
            districtFilter.innerHTML =
                '<option value="">All Districts</option>';

            // No state selected
            if (!selectedState) {

                districtFilter.disabled = true;

                return;
            }

            // Disable while loading
            districtFilter.disabled = true;

            // Loading option
            districtFilter.innerHTML =
                '<option value="">Loading districts...</option>';

            /*
             * Backend endpoint:
             * /api/districts/?state=STATE_NAME
             */

            const url =
                "/api/districts/?state=" +
                encodeURIComponent(selectedState);

            fetch(url, {
                method: "GET",
                headers: {
                    "X-Requested-With": "XMLHttpRequest"
                }
            })

            .then(function (response) {

                if (!response.ok) {
                    throw new Error(
                        "Unable to load districts."
                    );
                }

                return response.json();
            })

            .then(function (data) {

                districtFilter.innerHTML =
                    '<option value="">All Districts</option>';

                /*
                 * Support common API response formats:
                 *
                 * { districts: [...] }
                 * or
                 * [...]
                 */

                let districts = [];

                if (Array.isArray(data)) {

                    districts = data;

                } else if (Array.isArray(data.districts)) {

                    districts = data.districts;

                } else if (Array.isArray(data.results)) {

                    districts = data.results;
                }

                // Remove duplicates and empty values
                districts = [
                    ...new Set(
                        districts
                            .map(function (district) {

                                if (
                                    typeof district === "object" &&
                                    district !== null
                                ) {
                                    return (
                                        district.name ||
                                        district.district ||
                                        district.value ||
                                        ""
                                    );
                                }

                                return district;
                            })
                            .map(function (district) {
                                return String(district).trim();
                            })
                            .filter(Boolean)
                    )
                ];

                // Sort alphabetically
                districts.sort(function (a, b) {
                    return a.localeCompare(
                        b,
                        undefined,
                        {
                            sensitivity: "base"
                        }
                    );
                });

                districts.forEach(function (district) {

                    const option =
                        document.createElement("option");

                    option.value = district;
                    option.textContent = district;

                    districtFilter.appendChild(option);
                });

                districtFilter.disabled = false;

            })

            .catch(function (error) {

                console.error(
                    "Dataset district loading error:",
                    error
                );

                districtFilter.innerHTML =
                    '<option value="">Unable to load districts</option>';

                districtFilter.disabled = true;
            });

        });
    }


    /* ========================================================
       FORM SUBMISSION FEEDBACK
    ======================================================== */

    if (datasetForm) {

        datasetForm.addEventListener("submit", function () {

            const submitButton =
                datasetForm.querySelector(
                    'button[type="submit"]'
                );

            if (submitButton) {

                submitButton.disabled = true;

                submitButton.innerHTML =
                    '<span class="spinner-border spinner-border-sm me-1" ' +
                    'role="status" aria-hidden="true"></span> ' +
                    'Searching...';
            }

            // Flag this navigation as coming from the Search button
            // (as opposed to someone opening a filtered link directly),
            // so the results can be auto-scrolled to on the next load.
            try {
                sessionStorage.setItem("datasetJustSearched", "1");
            } catch (e) {
                // sessionStorage unavailable (e.g. private mode) — ignore.
            }

        });
    }


    /* ========================================================
       RESET BUTTON
       Scoped to the filter card specifically — a plain
       a[href*="dataset"] selector also matches the navbar's
       "Dataset" nav link, which sits earlier in the DOM and
       would otherwise steal this handler via querySelector.
    ======================================================== */

    const resetButton =
        datasetForm &&
        datasetForm.querySelector(
            'a[href*="dataset"]:not([href*="?"])'
        );

    if (resetButton) {

        resetButton.addEventListener(
            "click",
            function () {

                if (stateFilter) {
                    stateFilter.value = "";
                }

                if (districtFilter) {

                    districtFilter.innerHTML =
                        '<option value="">All Districts</option>';

                    districtFilter.value = "";
                    districtFilter.disabled = true;
                }

                try {
                    sessionStorage.removeItem("datasetJustSearched");
                } catch (e) {
                    // sessionStorage unavailable — ignore.
                }

            }
        );
    }


    /* ========================================================
       AUTO ENABLE DISTRICT WHEN STATE IS ALREADY SELECTED
       Useful when the page reloads after filtering.
    ======================================================== */

    if (
        stateFilter &&
        districtFilter &&
        stateFilter.value.trim() !== ""
    ) {

        districtFilter.disabled = false;
    }


    /* ========================================================
       PAGINATION LINKS — flag the same way as Search so paging
       through results also lands on the table, not the hero.
    ======================================================== */

    document
        .querySelectorAll(".dataset-pagination .page-link[href]")
        .forEach(function (link) {

            link.addEventListener("click", function () {

                try {
                    sessionStorage.setItem("datasetJustSearched", "1");
                } catch (e) {
                    // sessionStorage unavailable — ignore.
                }
            });
        });


    /* ========================================================
       SCROLL TO RECORDS AFTER FILTERING
       Auto-scrolls only when this page load resulted from the
       Search button, a pagination link, or Reset being clicked
       (flagged via sessionStorage just before navigation).
       Opening a filtered URL directly does not trigger the
       scroll — only a highlight.
    ======================================================== */

    const urlParams =
        new URLSearchParams(window.location.search);

    const hasFilters =
        urlParams.has("q") ||
        urlParams.has("state") ||
        urlParams.has("district") ||
        urlParams.has("year");

    if (hasFilters) {

        const recordsSection =
            document.querySelector(
                ".dataset-table-card, .dataset-empty-state"
            );

        let justSearched = false;

        try {
            justSearched =
                sessionStorage.getItem("datasetJustSearched") === "1";

            sessionStorage.removeItem("datasetJustSearched");
        } catch (e) {
            // sessionStorage unavailable — fall back to no auto-scroll.
        }

        if (recordsSection) {

            recordsSection.classList.add(
                "dataset-filtered-result"
            );

            setTimeout(function () {

                recordsSection.classList.remove(
                    "dataset-filtered-result"
                );

            }, 1800);

            if (justSearched) {

                const prefersReducedMotion =
                    window.matchMedia &&
                    window.matchMedia(
                        "(prefers-reduced-motion: reduce)"
                    ).matches;

                recordsSection.scrollIntoView({
                    behavior: prefersReducedMotion ? "auto" : "smooth",
                    block: "start"
                });
            }
        }
    }


    /* ========================================================
       SCROLL REVEAL (uses the site's existing .reveal-on-scroll /
       .is-revealed CSS mechanism from style.css — this only adds
       the class via IntersectionObserver. If the global site
       script already observes .reveal-on-scroll elements
       site-wide, this is redundant but harmless: both simply add
       the same "is-revealed" class once, and CSS ignores repeats.)
    ======================================================== */

    const revealTargets = document.querySelectorAll(
        ".reveal-on-scroll:not(.is-revealed)"
    );

    if (revealTargets.length && "IntersectionObserver" in window) {

        const revealObserver = new IntersectionObserver(
            function (entries, observer) {

                entries.forEach(function (entry) {

                    if (entry.isIntersecting) {

                        entry.target.classList.add("is-revealed");
                        observer.unobserve(entry.target);
                    }
                });
            },
            {
                threshold: 0.12,
                rootMargin: "0px 0px -40px 0px"
            }
        );

        revealTargets.forEach(function (target) {
            revealObserver.observe(target);
        });

    } else {

        // No IntersectionObserver support: reveal immediately.
        revealTargets.forEach(function (target) {
            target.classList.add("is-revealed");
        });
    }

});