/**
 * Shared shell helpers for Camera Intelligence System.
 * Mark the active top-nav link with data-nav matching the current page.
 */
(function () {
  function currentNavKey() {
    const path = window.location.pathname.replace(/\/+$/, "") || "/";
    if (path === "/" || path.endsWith("/features/overview")) return "home";
    // Engineer feature pages live under Site Admin → Others; highlight Site Admin
    if (
      path.includes("/features/zone-safety") ||
      path.includes("/features/vehicle") ||
      path.includes("/features/face") ||
      path.includes("/features/site-admin")
    ) {
      return "site-admin";
    }
    return "";
  }

  function ensureSiteAdminLink() {
    const nav = document.querySelector(".cis-nav-links");
    if (!nav || nav.querySelector('[data-nav="site-admin"]')) return;
    const a = document.createElement("a");
    a.href = "/features/site-admin/";
    a.setAttribute("data-nav", "site-admin");
    a.textContent = "Site Admin";
    nav.appendChild(a);
  }

  function highlightNav() {
    ensureSiteAdminLink();
    const key = currentNavKey();
    document.querySelectorAll(".cis-nav-links a[data-nav]").forEach((a) => {
      a.classList.toggle("active", a.getAttribute("data-nav") === key);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", highlightNav);
  } else {
    highlightNav();
  }

  window.CISShell = { currentNavKey, highlightNav };
})();
