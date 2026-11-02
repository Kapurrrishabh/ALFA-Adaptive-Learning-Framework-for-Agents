"use strict";
// Loaded in <head> after the stylesheet, so the saved or system theme is in place before the first paint.
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="theme-color"]').content = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim();
}
applyTheme(localStorage.getItem("si-theme") || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
