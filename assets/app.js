/**
 * Shared Application Script: Theme Toggle & Chart.js Reactivity
 * AI Student Performance & Academic Guidance System
 */
(function () {
  function initTheme() {
    var toggleBtn = document.getElementById('theme-toggle');
    if (!toggleBtn) return;

    toggleBtn.addEventListener('click', function () {
      var currentTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      var nextTheme = currentTheme === 'dark' ? 'light' : 'dark';

      document.documentElement.setAttribute('data-theme', nextTheme);
      try {
        localStorage.setItem('theme', nextTheme);
      } catch (e) {
        console.warn('Unable to access localStorage for theme preference', e);
      }

      // If Chart.js instances exist, update chart colors dynamically
      if (window.Chart && typeof Chart.instances === 'object') {
        var isDark = nextTheme === 'dark';
        Chart.defaults.color = isDark ? '#94A3B8' : '#334155';
        Chart.defaults.borderColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        Object.values(Chart.instances).forEach(function (inst) {
          try {
            inst.update();
          } catch (err) {
            // chart update error ignore
          }
        });
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTheme);
  } else {
    initTheme();
  }
})();
