/**
 * Shared shell helpers for Camera Intelligence System.
 * Mark the active top-nav link with data-nav matching the current page.
 */
(function () {
  function currentNavKey() {
    const path = window.location.pathname.replace(/\/+$/, "") || "/";
    if (path === "/" || path.endsWith("/features/overview")) return "overview";
    if (path.includes("/features/zone-safety")) return "zone-safety";
    if (path.includes("/features/vehicle")) return "vehicle";
    if (path.includes("/features/face")) return "face";
    return "";
  }

  function highlightNav() {
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
