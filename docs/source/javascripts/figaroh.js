/* FIGAROH docs — interactive pieces.
 *
 * Every widget enhances markup that is already readable without JS:
 *   - [data-fg-copy]        copy a string, show "Copied"
 *   - .fg-codecard          landing code tabs
 *   - .fg-tasks             landing "What do you want to do?" picker
 *   - .fg-guide             workflow guide: stepper + "starting from" toggle
 *   - .fg-filters           prompt-library filter chips
 *   - .fg-copy-md           copy this page as Markdown (from llms.txt output)
 *
 * Runs on every page load, including Material's instant navigation.
 */
(function() {
'use strict';

function copyText(text, button) {
  var done = function() {
    var label = button.querySelector('[data-fg-copy-label]') || button;
    var before = label.textContent;
    label.textContent = 'Copied';
    setTimeout(function() {
      label.textContent = before;
    }, 1600);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, function() {});
  }
}

function initCopyButtons(root) {
  root.querySelectorAll('[data-fg-copy]').forEach(function(btn) {
    btn.addEventListener('click', function() {
      copyText(btn.getAttribute('data-fg-copy'), btn);
    });
  });
  root.querySelectorAll('.fg-prompt').forEach(function(card) {
    var btn = card.querySelector('.fg-copy');
    var text = card.querySelector('.fg-prompt__text');
    if (btn && text)
      btn.addEventListener('click', function() {
        copyText(text.textContent.trim(), btn);
      });
  });
}

function initTabs(root) {
  root.querySelectorAll('.fg-codecard').forEach(function(card) {
    var tabs = card.querySelectorAll('[role=tab]');
    var panels = card.querySelectorAll('.fg-codecard__panel');
    tabs.forEach(function(tab, i) {
      tab.addEventListener('click', function() {
        tabs.forEach(function(t, j) {
          t.setAttribute('aria-selected', String(i === j));
        });
        panels.forEach(function(p, j) {
          p.hidden = i !== j;
        });
      });
    });
  });
}

function initTaskPicker(root) {
  root.querySelectorAll('.fg-tasks').forEach(function(group) {
    var buttons = group.querySelectorAll('.fg-task');
    buttons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        var id = btn.getAttribute('data-task');
        buttons.forEach(function(b) {
          b.setAttribute('aria-pressed', String(b === btn));
        });
        root.querySelectorAll('.fg-detail').forEach(function(d) {
          d.hidden = d.getAttribute('data-task') !== id;
        });
      });
    });
  });
}

/* Workflow guide.
 * Markup: <div class="fg-guide"> containing <div class="fg-step" data-title=…>
 * blocks; inside them, blocks marked data-fg-path="example|own".
 * Without JS the page reads top to bottom with both paths labelled. */
var PATH_KEY = 'figaroh.guide.path';

function storageGet(k) {
  try {
    return window.localStorage.getItem(k);
  } catch (e) {
    return null;
  }
}
function storageSet(k, v) {
  try {
    window.localStorage.setItem(k, v);
  } catch (e) { /* private mode */
  }
}

function initGuide(root) {
  root.querySelectorAll('.fg-guide').forEach(function(guide) {
    var steps =
        Array.prototype.slice.call(guide.querySelectorAll(':scope > .fg-step'));
    if (!steps.length) return;
    document.documentElement.classList.add('fg-js');

    var state = {step: 0, path: storageGet(PATH_KEY) || 'example', all: false};

    var hash = window.location.hash.slice(1);
    steps.forEach(function(s, i) {
      if (hash && (s.id === hash || s.querySelector('[id=\'' + hash + '\']')))
        state.step = i;
    });

    // Bar: starting-from toggle, show-all toggle.
    var bar = document.createElement('div');
    bar.className = 'fg-guide-bar';
    bar.innerHTML = '<span class="fg-guide-bar__label">Starting from</span>' +
        '<span class="fg-seg" role="group" aria-label="Starting from">' +
        '<button type="button" data-path="example">An example robot</button>' +
        '<button type="button" data-path="own">My own robot</button></span>' +
        '<span class="fg-guide-bar__spacer"></span>' +
        '<span class="fg-seg"><button type="button" data-all>Show all steps</button></span>';
    guide.parentNode.insertBefore(bar, guide);

    // Stepper.
    var stepper = document.createElement('ol');
    stepper.className = 'fg-stepper';
    stepper.setAttribute('aria-label', 'Steps');
    steps.forEach(function(s, i) {
      var li = document.createElement('li');
      var b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = '<span class="fg-stepper__dot">' + (i + 1) +
          '</span><span class="fg-stepper__name"></span>';
      b.querySelector('.fg-stepper__name').textContent =
          s.getAttribute('data-title') || ('Step ' + (i + 1));
      b.addEventListener('click', function() {
        state.step = i;
        state.all = false;
        render(true);
      });
      li.appendChild(b);
      stepper.appendChild(li);
    });
    guide.parentNode.insertBefore(stepper, guide);

    // Prev / next under each step.
    steps.forEach(function(s, i) {
      var nav = document.createElement('div');
      nav.className = 'fg-step-nav';
      if (i > 0) {
        var p = document.createElement('button');
        p.type = 'button';
        p.className = 'fg-btn';
        p.textContent =
            '← ' + (steps[i - 1].getAttribute('data-title') || 'Previous');
        p.addEventListener('click', function() {
          state.step = i - 1;
          render(true);
        });
        nav.appendChild(p);
      } else {
        nav.appendChild(document.createElement('span'));
      }
      if (i < steps.length - 1) {
        var n = document.createElement('button');
        n.type = 'button';
        n.className = 'fg-btn fg-btn--primary';
        n.textContent =
            'Next: ' + (steps[i + 1].getAttribute('data-title') || 'step') +
            ' →';
        n.addEventListener('click', function() {
          state.step = i + 1;
          render(true);
        });
        nav.appendChild(n);
      }
      nav.setAttribute('data-fg-stepnav', '');
      s.appendChild(nav);
    });

    bar.querySelectorAll('[data-path]').forEach(function(b) {
      b.addEventListener('click', function() {
        state.path = b.getAttribute('data-path');
        storageSet(PATH_KEY, state.path);
        render(false);
      });
    });
    var allBtn = bar.querySelector('[data-all]');
    allBtn.addEventListener('click', function() {
      state.all = !state.all;
      render(false);
    });

    function render(scroll) {
      steps.forEach(function(s, i) {
        s.hidden = !state.all && i !== state.step;
        var nav = s.querySelector('[data-fg-stepnav]');
        if (nav) nav.hidden = state.all;
      });
      stepper.querySelectorAll('button').forEach(function(b, i) {
        if (!state.all && i === state.step)
          b.setAttribute('aria-current', 'step');
        else
          b.removeAttribute('aria-current');
        b.classList.toggle('is-past', !state.all && i < state.step);
      });
      guide.querySelectorAll('[data-fg-path]').forEach(function(el) {
        el.hidden = el.getAttribute('data-fg-path') !== state.path;
      });
      bar.querySelectorAll('[data-path]').forEach(function(b) {
        b.setAttribute(
            'aria-pressed', String(b.getAttribute('data-path') === state.path));
      });
      allBtn.setAttribute('aria-pressed', String(state.all));
      allBtn.textContent = state.all ? 'One step at a time' : 'Show all steps';
      if (scroll) {
        var id = steps[state.step].id;
        if (id && history.replaceState)
          history.replaceState(null, '', '#' + id);
        bar.scrollIntoView({behavior: 'smooth', block: 'start'});
      }
    }
    render(false);
  });
}

function initFilters(root) {
  root.querySelectorAll('.fg-filters').forEach(function(bar) {
    var target = document.getElementById(bar.getAttribute('data-target'));
    if (!target) return;
    var cards = target.querySelectorAll('[data-cat]');
    bar.querySelectorAll('button').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var cat = btn.getAttribute('data-filter');
        bar.querySelectorAll('button').forEach(function(b) {
          b.setAttribute('aria-pressed', String(b === btn));
        });
        cards.forEach(function(c) {
          c.hidden = cat !== 'all' && c.getAttribute('data-cat') !== cat;
        });
      });
    });
  });
}

/* mkdocs-llmstxt writes <page>/index.md next to each page's index.html. */
function initCopyMarkdown(root) {
  root.querySelectorAll('.fg-copy-md').forEach(function(btn) {
    btn.addEventListener('click', function(ev) {
      ev.preventDefault();
      var url = btn.getAttribute('href');
      fetch(url)
          .then(function(r) {
            if (!r.ok) throw new Error(r.status);
            return r.text();
          })
          .then(
              function(md) {
                copyText(md, btn);
              },
              function() {
                window.open(url, '_blank');
              });
    });
  });
}

function init() {
  var root = document;
  initCopyButtons(root);
  initTabs(root);
  initTaskPicker(root);
  initGuide(root);
  initFilters(root);
  initCopyMarkdown(root);
}

if (window.document$ && typeof window.document$.subscribe === 'function') {
  window.document$.subscribe(init);
} else if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
})();
