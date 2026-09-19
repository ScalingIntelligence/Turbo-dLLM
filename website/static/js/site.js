(function () {
  'use strict';
  var root = document.documentElement;
  root.classList.add('js');
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function copyText(text, el, cls) {
    function done() {
      el.classList.add(cls);
      setTimeout(function () { el.classList.remove(cls); }, 1600);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else {
      fallback();
    }
    function fallback() {
      var ta = document.createElement('textarea');
      ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'absolute'; ta.style.left = '-9999px';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ }
      document.body.removeChild(ta);
    }
  }

  // Install command
  var install = document.getElementById('install-btn');
  if (install) {
    install.addEventListener('click', function () { copyText(install.getAttribute('data-copy'), install, 'copied'); });
  }

  // BibTeX
  var bibBtn = document.getElementById('copy-bib');
  var bib = document.getElementById('bibtex');
  if (bibBtn && bib) {
    bibBtn.addEventListener('click', function () {
      copyText(bib.textContent, bibBtn, 'copied');
      bibBtn.textContent = 'Copied';
      setTimeout(function () { bibBtn.textContent = 'Copy'; }, 1600);
    });
  }

  // Reveal figure cards on scroll
  var reveals = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && !reduceMotion) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    Array.prototype.forEach.call(reveals, function (el) { io.observe(el); });
  } else {
    Array.prototype.forEach.call(reveals, function (el) { el.classList.add('in'); });
  }

  // Left section-progress nav
  var nav = document.querySelector('.side-nav');
  if (nav) {
    var links = {};
    Array.prototype.forEach.call(nav.querySelectorAll('a[data-target]'), function (a) {
      links[a.getAttribute('data-target')] = a;
    });
    var sections = Object.keys(links).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var trigger = document.getElementById('overview');
    var onScroll = function () {
      nav.classList.toggle('visible', trigger ? trigger.getBoundingClientRect().top <= 120 : window.scrollY > 400);
      var best = sections[0], bestTop = -Infinity;
      sections.forEach(function (el) {
        var top = el.getBoundingClientRect().top;
        if (top - 160 <= 0 && top > bestTop) { bestTop = top; best = el; }
      });
      Object.keys(links).forEach(function (k) { links[k].classList.toggle('active', best && best.id === k); });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    onScroll();
  }
})();
