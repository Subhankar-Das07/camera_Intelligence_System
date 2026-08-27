/**
 * Landing dummy gate — no credentials required.
 * sessionStorage key: cis_demo_gate
 */
(function () {
  const GATE_KEY = "cis_demo_gate";
  const SITE_ADMIN = "/features/site-admin/";

  const form = document.getElementById("lp-login-form");
  const gate = document.getElementById("lp-gate");
  const resetBtn = document.getElementById("lp-reset");

  function hasGate() {
    try {
      return sessionStorage.getItem(GATE_KEY) === "1";
    } catch (_) {
      return false;
    }
  }

  function setGate() {
    try {
      sessionStorage.setItem(GATE_KEY, "1");
    } catch (_) {
      /* private mode — still navigate */
    }
  }

  function clearGate() {
    try {
      sessionStorage.removeItem(GATE_KEY);
    } catch (_) {
      /* ignore */
    }
  }

  function syncUI() {
    if (!form || !gate) return;
    const on = hasGate();
    form.hidden = on;
    gate.hidden = !on;
  }

  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    setGate();
    window.location.href = SITE_ADMIN;
  });

  resetBtn?.addEventListener("click", () => {
    clearGate();
    syncUI();
    form?.querySelector('input[name="username"]')?.focus();
  });

  syncUI();
})();
