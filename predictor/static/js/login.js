/* ============================================================
   LOGIN — behavior
   Hybrid CNN-LSTM Crime Prediction Framework
   Vanilla ES6, no external deps beyond Bootstrap Icons markup.
   ============================================================ */

(() => {
  "use strict";

  const form = document.getElementById("loginForm");
  if (!form) return;

  const emailInput = document.getElementById("id_email");
  const emailError = document.getElementById("emailError");

  const passwordInput = document.getElementById("id_password");
  const passwordError = document.getElementById("passwordError");

  const submitBtn = document.getElementById("loginSubmitBtn");

  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  /* ---------- Show / hide password ---------- */
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

  /* ---------- Field-level validation ---------- */
  function setFieldState(inputEl, errorEl, valid, hasValue) {
    inputEl.classList.remove("is-valid", "is-invalid");
    if (errorEl) errorEl.classList.remove("show");

    if (!hasValue) return; // don't show state until user interacts

    if (valid) {
      inputEl.classList.add("is-valid");
    } else {
      inputEl.classList.add("is-invalid");
      if (errorEl) errorEl.classList.add("show");
    }
  }

  function validateEmail() {
    const value = emailInput.value.trim();
    const valid = EMAIL_RE.test(value);
    setFieldState(emailInput, emailError, valid, value.length > 0);
    return valid;
  }

  function validatePassword() {
    const value = passwordInput.value;
    const valid = value.length > 0;
    setFieldState(passwordInput, passwordError, valid, document.activeElement !== passwordInput);
    return valid;
  }

  emailInput.addEventListener("input", validateEmail);
  emailInput.addEventListener("blur", validateEmail);
  passwordInput.addEventListener("blur", validatePassword);

  /* ---------- Submit handling ---------- */
  form.addEventListener("submit", (e) => {
    const emailValid = validateEmail();
    passwordInput.classList.remove("is-invalid");
    passwordError.classList.remove("show");
    const passwordValid = passwordInput.value.length > 0;

    if (!passwordValid) {
      passwordInput.classList.add("is-invalid");
      passwordError.classList.add("show");
    }

    if (!emailValid || !passwordValid) {
      e.preventDefault();
      return;
    }

    // Loading state — Django handles the actual redirect/response on submit
    submitBtn.classList.add("is-loading");
    submitBtn.disabled = true;
  });
})();