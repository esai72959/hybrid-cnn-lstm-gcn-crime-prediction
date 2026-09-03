/* ============================================================
   SIGNUP — behavior
   Hybrid CNN-LSTM Crime Prediction Framework
   Vanilla ES6, no external deps.
   ============================================================ */

(() => {
  "use strict";

  const form = document.getElementById("signupForm");
  if (!form) return;

  const fullNameInput = document.getElementById("id_full_name");
  const fullNameError = document.getElementById("fullNameError");

  const emailInput = document.getElementById("id_email");
  const emailError = document.getElementById("emailError");

  const mobileInput = document.getElementById("id_mobile");
  const mobileError = document.getElementById("mobileError");

  const passwordInput = document.getElementById("id_password");
  const passwordError = document.getElementById("passwordError");
  const reqList = document.getElementById("reqList");
  const strengthBars = document.querySelectorAll(".pw-strength-bar");
  const strengthLabel = document.querySelector(".pw-strength-label .val");

  const confirmInput = document.getElementById("id_confirm_password");
  const confirmMatchMsg = document.getElementById("confirmMatchMsg");

  const termsCheckbox = document.getElementById("id_terms");
  const submitBtn = document.getElementById("signupSubmitBtn");

  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const MOBILE_RE = /^[0-9]{10}$/;

  function setFieldState(inputEl, errorEl, valid, hasValue) {
    inputEl.classList.remove("is-valid", "is-invalid");
    if (errorEl) errorEl.classList.remove("show");
    if (!hasValue) return;

    if (valid) {
      inputEl.classList.add("is-valid");
    } else {
      inputEl.classList.add("is-invalid");
      if (errorEl) errorEl.classList.add("show");
    }
  }

  /* ---------- Show / hide password (both fields) ---------- */
  document.querySelectorAll(".auth-toggle-visibility").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const input = document.getElementById(targetId);
      if (!input) return;

      const isHidden = input.type === "password";
      input.type = isHidden ? "text" : "password";

      const icon = btn.querySelector("i");
      icon.classList.toggle("bi-eye", !isHidden);
      icon.classList.toggle("bi-eye-slash", isHidden);
      btn.setAttribute("aria-label", isHidden ? "Hide password" : "Show password");
    });
  });

  /* ---------- Full name ---------- */
  function validateFullName() {
    const value = fullNameInput.value.trim();
    const valid = value.length >= 2;
    setFieldState(fullNameInput, fullNameError, valid, value.length > 0);
    return valid;
  }

  /* ---------- Email ---------- */
  function validateEmail() {
    const value = emailInput.value.trim();
    const valid = EMAIL_RE.test(value);
    setFieldState(emailInput, emailError, valid, value.length > 0);
    return valid;
  }

  /* ---------- Mobile ---------- */
  function validateMobile() {
    const value = mobileInput.value.trim();
    const valid = MOBILE_RE.test(value);
    setFieldState(mobileInput, mobileError, valid, value.length > 0);
    return valid;
  }

  /* ---------- Password strength ---------- */
  function evaluatePassword(value) {
    const rules = {
      length: value.length >= 8,
      upper: /[A-Z]/.test(value),
      number: /[0-9]/.test(value),
      special: /[^A-Za-z0-9]/.test(value),
    };

    reqList.querySelectorAll("li").forEach((li) => {
      const rule = li.getAttribute("data-rule");
      const met = rules[rule];
      li.classList.toggle("met", met);
      const icon = li.querySelector("i");
      icon.classList.toggle("bi-circle", !met);
      icon.classList.toggle("bi-check-circle-fill", met);
    });

    const metCount = Object.values(rules).filter(Boolean).length;

    let level = 0;
    let label = "—";
    if (value.length === 0) {
      level = 0;
      label = "—";
    } else if (metCount <= 1) {
      level = 1;
      label = "Weak";
    } else if (metCount === 2) {
      level = 2;
      label = "Fair";
    } else if (metCount === 3) {
      level = 3;
      label = "Good";
    } else {
      level = 4;
      label = "Strong";
    }

    strengthBars.forEach((bar, i) => {
      bar.classList.toggle("is-active", i < level);
    });
    strengthLabel.textContent = label;

    return metCount === 4;
  }

  function validatePassword() {
    const value = passwordInput.value;
    const strong = evaluatePassword(value);
    setFieldState(passwordInput, passwordError, strong, value.length > 0 && document.activeElement !== passwordInput);
    return value.length > 0 && strong;
  }

  /* ---------- Confirm password ---------- */
  function validateConfirm() {
    const value = confirmInput.value;
    if (value.length === 0) {
      confirmMatchMsg.classList.remove("show", "match", "no-match");
      confirmInput.classList.remove("is-valid", "is-invalid");
      return false;
    }

    const matches = value === passwordInput.value;
    confirmMatchMsg.classList.add("show");
    confirmMatchMsg.classList.toggle("match", matches);
    confirmMatchMsg.classList.toggle("no-match", !matches);
    confirmMatchMsg.querySelector("i").className = matches ? "bi bi-check-circle" : "bi bi-x-circle";
    confirmMatchMsg.querySelector("span").textContent = matches ? "Passwords match" : "Passwords do not match";

    confirmInput.classList.toggle("is-valid", matches);
    confirmInput.classList.toggle("is-invalid", !matches);

    return matches;
  }

  fullNameInput.addEventListener("input", validateFullName);
  emailInput.addEventListener("input", validateEmail);
  mobileInput.addEventListener("input", validateMobile);
  passwordInput.addEventListener("input", () => {
    evaluatePassword(passwordInput.value);
    if (confirmInput.value.length > 0) validateConfirm();
  });
  passwordInput.addEventListener("blur", validatePassword);
  confirmInput.addEventListener("input", validateConfirm);

  /* ---------- Submit ---------- */
  form.addEventListener("submit", (e) => {
    const nameValid = validateFullName();
    const emailValid = validateEmail();
    const mobileValid = validateMobile();
    const passwordValid = validatePassword();
    const confirmValid = validateConfirm();
    const termsChecked = termsCheckbox.checked;

    if (!termsChecked) {
      termsCheckbox.focus();
    }

    if (!nameValid || !emailValid || !mobileValid || !passwordValid || !confirmValid || !termsChecked) {
      e.preventDefault();
      return;
    }

    submitBtn.classList.add("is-loading");
    submitBtn.disabled = true;
  });
})();