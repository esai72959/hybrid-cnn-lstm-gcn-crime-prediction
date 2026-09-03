/**
 * Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction
 * Global Application JavaScript Engine (script.js)
 * 
 * Architecture: ES6 Modular / Event-Driven Architecture
 * Scope: Global Navigation, Theme Engine, Global UI Animations, and Shared Utilities.
 */

'use strict';

// Initialize Global Namespace
window.CPS = window.CPS || {};

document.addEventListener('DOMContentLoaded', () => {
    CPS.PageLoader.init();
    CPS.NavbarEngine.init();
    CPS.MobileNav.init();
    CPS.ThemeEngine.init();
    CPS.BackToTop.init();
    CPS.ScrollRevealEngine.init();
});

/* ==========================================================================
   1. PAGE LOADER MODULE
   ========================================================================== */
CPS.PageLoader = (() => {
    const loader = document.querySelector('.page-loader');

    const fadeOut = () => {
        let opacity = 1;
        const fade = () => {
            opacity -= 0.05;
            if (opacity <= 0) {
                loader.style.opacity = '0';
                loader.style.display = 'none';
                if (loader.parentNode) {
                    loader.parentNode.removeChild(loader);
                }
            } else {
                loader.style.opacity = opacity.toString();
                requestAnimationFrame(fade);
            }
        };
        requestAnimationFrame(fade);
    };

    const hide = () => {
        document.body.classList.add('is-loaded');
        if (loader) {
            loader.classList.add('is-hidden');
            fadeOut();
        }
    };

    const init = () => {
    hide();
};

    return { init };
})();

/* ==========================================================================
   2. NAVBAR ENGINE MODULE
   ========================================================================== */
CPS.NavbarEngine = (() => {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelectorAll('.nav-link');

    const handleScroll = () => {
        if (!navbar) return;
        if (window.scrollY > 20) {
            navbar.classList.add('navbar-scrolled');
        } else {
            navbar.classList.remove('navbar-scrolled');
        }
    };

    const normalizeUrl = (path) => {
        if (!path) return '';
        let clean = path.split('?')[0].split('#')[0];
        if (!clean.startsWith('/')) clean = '/' + clean;
        if (!clean.endsWith('/')) clean = clean + '/';
        return clean.toLowerCase();
    };

    const updateActiveState = () => {
        const currentNormalizedPath = normalizeUrl(window.location.pathname);
        
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (!href || href === '#') return;

            const hrefNormalizedPath = normalizeUrl(href);
            
            if (currentNormalizedPath === hrefNormalizedPath) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    };

    const init = () => {
        handleScroll();
        updateActiveState();
        window.addEventListener('scroll', handleScroll, { passive: true });
    };

    return { init };
})();

/* ==========================================================================
   3. MOBILE NAVIGATION MODULE
   ========================================================================== */
CPS.MobileNav = (() => {
    const toggleBtn = document.querySelector('.navbar-toggler');
    const navMenu = document.querySelector('.navbar-collapse');
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');

    const closeMenu = () => {
        if (navMenu && navMenu.classList.contains('show')) {
            if (toggleBtn && typeof bootstrap !== 'undefined') {
                const bsCollapse = bootstrap.Collapse.getInstance(navMenu) || new bootstrap.Collapse(navMenu, { toggle: false });
                bsCollapse.hide();
            } else {
                navMenu.classList.remove('show');
            }
        }
    };

    const handleOutsideClick = (e) => {
        if (!navMenu || !toggleBtn) return;
        const isClickInsideMenu = navMenu.contains(e.target);
        const isClickOnToggle = toggleBtn.contains(e.target);
        
        if (!isClickInsideMenu && !isClickOnToggle && navMenu.classList.contains('show')) {
            closeMenu();
        }
    };

    const init = () => {
        if (!toggleBtn || !navMenu) return;

        document.addEventListener('click', handleOutsideClick);

        navLinks.forEach(link => {
            link.addEventListener('click', closeMenu);
        });
    };

    return { init };
})();

/* ==========================================================================
   4. THEME ENGINE MODULE
   ========================================================================== */
CPS.ThemeEngine = (() => {
    const STORAGE_KEY = 'cps_theme_preference';
    const themeToggleBtns = document.querySelectorAll('.theme-toggle-btn');

    const getStoredTheme = () => localStorage.getItem(STORAGE_KEY);
    const setStoredTheme = (theme) => localStorage.setItem(STORAGE_KEY, theme);

    const getPreferredTheme = () => {
        const stored = getStoredTheme();
        if (stored) return stored;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    };

    const applyTheme = (theme) => {
        document.documentElement.setAttribute('data-bs-theme', theme);
        document.documentElement.setAttribute('data-theme', theme);
        
        themeToggleBtns.forEach(btn => {
            if (theme === 'dark') {
                btn.classList.add('is-dark');
                btn.setAttribute('aria-label', 'Switch to light mode');
            } else {
                btn.classList.remove('is-dark');
                btn.setAttribute('aria-label', 'Switch to dark mode');
            }
        });
    };

    const toggleTheme = () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || getPreferredTheme();
        const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setStoredTheme(nextTheme);
        applyTheme(nextTheme);
    };

    const init = () => {
        const initialTheme = getPreferredTheme();
        applyTheme(initialTheme);

        themeToggleBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                toggleTheme();
            });
        });

        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!getStoredTheme()) {
                applyTheme(e.matches ? 'dark' : 'light');
            }
        });
    };

    return { init, applyTheme, toggleTheme };
})();

/* ==========================================================================
   5. BACK TO TOP MODULE
   ========================================================================== */
CPS.BackToTop = (() => {
    const backToTopBtn = document.getElementById('backToTopBtn') || document.querySelector('.back-to-top');

    const handleScroll = () => {
        if (!backToTopBtn) return;
        if (window.scrollY > 300) {
            backToTopBtn.classList.add('is-visible');
        } else {
            backToTopBtn.classList.remove('is-visible');
        }
    };

    const scrollToTop = (e) => {
        if (e) e.preventDefault();
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    };

    const init = () => {
        if (!backToTopBtn) return;
        window.addEventListener('scroll', handleScroll, { passive: true });
        backToTopBtn.addEventListener('click', scrollToTop);
        handleScroll();
    };

    return { init };
})();

/* ==========================================================================
   6. SCROLL REVEAL MODULE (SINGLE INTERSECTION OBSERVER)
   ========================================================================== */
CPS.ScrollRevealEngine = (() => {
    let observer = null;

    const init = () => {
        const revealElements = document.querySelectorAll('.reveal-on-scroll');
        if (!revealElements.length) return;

        if (!('IntersectionObserver' in window)) {
            revealElements.forEach(el => el.classList.add('is-revealed'));
            return;
        }

        observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-revealed');
                    obs.unobserve(entry.target);
                }
            });
        }, {
            root: null,
            threshold: 0.15,
            rootMargin: '0px 0px -50px 0px'
        });

        revealElements.forEach(el => observer.observe(el));
    };

    return { init };
})();

/* ==========================================================================
   7. SHARED COUNTER UTILITY MODULE (SINGLE SHARED OBSERVER)
   ========================================================================== */
CPS.animateCounters = (() => {
    let sharedCounterObserver = null;

    const runAnimation = (counterEl) => {
        if (counterEl.dataset.counterAnimated === 'true') return;
        counterEl.dataset.counterAnimated = 'true';

        const targetAttr =counterEl.getAttribute('data-counter') || counterEl.getAttribute('data-target') || counterEl.textContent.trim();
        const duration = parseInt(counterEl.getAttribute('data-duration'), 10) || 2000;
        const suffix = counterEl.getAttribute('data-suffix') || '';
        const prefix = counterEl.getAttribute('data-prefix') || '';

        const parsedTarget = parseFloat(targetAttr.replace(/,/g, ''));
        if (isNaN(parsedTarget)) return;

        const decimalMatch = targetAttr.match(/\.(\d+)/);
        const decimals = decimalMatch ? decimalMatch[1].length : 0;

        let startTime = null;
        const startValue = 0;

        const step = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            
            // Ease-out quad formula for natural deceleration
            const easeOutProgress = 1 - (1 - progress) * (1 - progress);
            const currentValue = startValue + (parsedTarget - startValue) * easeOutProgress;

            counterEl.textContent = `${prefix}${currentValue.toFixed(decimals)}${suffix}`;

            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                counterEl.textContent = `${prefix}${parsedTarget.toFixed(decimals)}${suffix}`;
            }
        };

        window.requestAnimationFrame(step);
    };

    const getObserver = () => {
        if (!sharedCounterObserver && ('IntersectionObserver' in window)) {
            sharedCounterObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        runAnimation(entry.target);
                        observer.unobserve(entry.target);
                    }
                });
            }, {
                threshold: 0.2
            });
        }
        return sharedCounterObserver;
    };

    return (selector = '.stat-counter') => {
        const counterElements = document.querySelectorAll(selector);
        if (!counterElements.length) return;

        const observer = getObserver();

        if (!observer) {
            counterElements.forEach(el => runAnimation(el));
            return;
        }

        counterElements.forEach(el => {
            if (el.dataset.counterAnimated !== 'true') {
                observer.observe(el);
            }
        });
    };
})();