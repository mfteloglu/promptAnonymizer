/* global chrome */
(function () {
  const API_URL = 'http://127.0.0.1:8000/anonymize_text';

  // Utility: debounce
  const debounce = (fn, wait) => {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), wait);
    };
  };

  // Create/hide overlay and modal
  function createOverlay() {
    const overlay = document.createElement('div');
    overlay.id = 'prompt-anonymizer-overlay';
    overlay.className = 'prompt-anonymizer-overlay hidden';
    overlay.innerHTML = `
      <div class="prompt-anonymizer-overlay-backdrop"></div>
      <div class="prompt-anonymizer-modal">
        <div class="prompt-anonymizer-spinner" id="prompt-anonymizer-loading" aria-label="Loading"></div>
        <div class="prompt-anonymizer-modal-content hidden" id="prompt-anonymizer-modal-content">
          <h3 class="prompt-anonymizer-modal-title">Anonymized Prompt</h3>
          <div class="prompt-anonymizer-modal-body">
            <textarea id="prompt-anonymizer-final-text" class="prompt-anonymizer-textarea" rows="10"></textarea>
          </div>
          <div class="prompt-anonymizer-modal-actions">
            <button id="prompt-anonymizer-accept" class="prompt-anonymizer-btn prompt-anonymizer-btn-primary">Accept</button>
            <button id="prompt-anonymizer-cancel" class="prompt-anonymizer-btn prompt-anonymizer-btn-secondary">Cancel</button>
          </div>
          <div id="prompt-anonymizer-error" class="prompt-anonymizer-error hidden"></div>
        </div>
      </div>`;
    document.documentElement.appendChild(overlay);
    return overlay;
  }

  function showLoading() {
    const overlay = document.getElementById('prompt-anonymizer-overlay') || createOverlay();
    overlay.classList.remove('hidden');
    overlay.querySelector('#prompt-anonymizer-loading')?.classList.remove('hidden');
    overlay.querySelector('#prompt-anonymizer-modal-content')?.classList.add('hidden');
  }

  function showResult(finalText, errorMsg, sensitivityInfo) {
    const overlay = document.getElementById('prompt-anonymizer-overlay') || createOverlay();
    overlay.classList.remove('hidden');
    overlay.querySelector('#prompt-anonymizer-loading')?.classList.add('hidden');
    const content = overlay.querySelector('#prompt-anonymizer-modal-content');
    const textarea = overlay.querySelector('#prompt-anonymizer-final-text');
    const error = overlay.querySelector('#prompt-anonymizer-error');
    
    // Handle sensitivity warning
    let warningEl = overlay.querySelector('.prompt-anonymizer-sensitivity-warning');
    if (sensitivityInfo && sensitivityInfo.is_sensitive) {
      if (!warningEl) {
        warningEl = document.createElement('div');
        warningEl.className = 'prompt-anonymizer-sensitivity-warning';
        // Insert before textarea container (modal body)
        const body = overlay.querySelector('.prompt-anonymizer-modal-body');
        body.parentNode.insertBefore(warningEl, body);
      }
      warningEl.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 4px;">
          <div><strong>⚠️ Warning: Sensitive Content Detected</strong></div>
          <div style="font-size: 0.9em;">Type: ${sensitivityInfo.context_type}</div>
          <div style="font-size: 0.9em; opacity: 0.9;">${sensitivityInfo.explanation || ''}</div>
        </div>`;
      warningEl.classList.remove('hidden');
    } else {
      if (warningEl) warningEl.classList.add('hidden');
    }

    if (textarea) textarea.value = finalText || '';
    if (error) {
      if (errorMsg) {
        error.textContent = errorMsg;
        error.classList.remove('hidden');
      } else {
        error.textContent = '';
        error.classList.add('hidden');
      }
    }
    content?.classList.remove('hidden');
  }

  function hideOverlay() {
    const overlay = document.getElementById('prompt-anonymizer-overlay');
    if (overlay) overlay.classList.add('hidden');
  }

  // Messaging to background to avoid page CORS/CSP issues
  function anonymize(text) {
    return new Promise((resolve) => {
      try {
        const port = chrome.runtime.connect({name: "prompt_anonymizer_anonymize"});
        
        // Set a long timeout (e.g., 5 minutes)
        const timeout = setTimeout(() => {
            port.disconnect();
            resolve({ ok: false, error: "Timeout waiting for response (5m)" });
        }, 300000);

        port.onMessage.addListener((msg) => {
            clearTimeout(timeout);
            resolve(msg);
            port.disconnect();
        });

        port.onDisconnect.addListener(() => {
             clearTimeout(timeout);
             if (chrome.runtime.lastError) {
                 resolve({ ok: false, error: chrome.runtime.lastError.message });
             } else {
                 // Disconnected without message usually means background script died or closed port
                 // But if we resolved already, this is fine.
                 // If not resolved, we should resolve with error.
                 // However, we can't easily check if resolved.
                 // But since we resolve in onMessage, this is mostly for errors.
             }
        });

        port.postMessage({ type: 'ANONYMIZE_TEXT', url: API_URL, text });
        
      } catch (e) {
        resolve({ ok: false, error: String(e) });
      }
    });
  }

  // Find ChatGPT's prompt input (textarea or contenteditable)
  function getPromptInput() {
    const selectors = [
      'textarea[data-testid="prompt-textarea"]',
      'textarea[aria-label]',
      'textarea',
      'div[contenteditable="true"][data-testid="prompt-textarea"]',
      'div[contenteditable="true"][role="textbox"]',
      'div[contenteditable="true"]'
    ];
    const nodes = selectors.flatMap(sel => Array.from(document.querySelectorAll(sel)));
    const visible = nodes.filter(el => el && el.offsetParent !== null);
    visible.sort((a, b) => (b.clientWidth * b.clientHeight) - (a.clientWidth * a.clientHeight));
    return visible[0] || null;
  }

  function readInputValue(el) {
    if (!el) return '';
    if (el.tagName === 'TEXTAREA') return (el.value || '').trim();
    if (el.getAttribute('contenteditable') === 'true') return (el.innerText || el.textContent || '').trim();
    return '';
  }

  function setInputValue(el, value) {
    if (!el) return;
    if (el.tagName === 'TEXTAREA') {
      el.value = value;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      return;
    }
    if (el.getAttribute('contenteditable') === 'true') {
      el.textContent = value;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('keyup', { bubbles: true }));
    }
  }

  function getSendButton() {
    const testIdBtn = document.querySelector('button[data-testid="send-button"]');
    if (testIdBtn && testIdBtn.offsetParent !== null) return testIdBtn;
    const input = getPromptInput();
    if (!input) return null;
    const form = input.closest('form');
    if (form) {
      const btn = form.querySelector('button[type="submit"], button');
      if (btn && btn.offsetParent !== null) return btn;
    }
    const buttons = Array.from(document.querySelectorAll('button'));
    const vis = buttons.filter((b) => b.offsetParent !== null);
    return vis[vis.length - 1] || null;
  }

  let intercepting = false;
  // When true, allow the very next submit to pass through unmodified.
  let skipNextIntercept = false;
  let lastAcceptedValue = '';

  async function handleInterceptSubmit(source) {
    // If we flagged to skip, clear flag and allow normal submission
    if (skipNextIntercept) {
      skipNextIntercept = false;
      return; // do nothing; page's native handler will send
    }
    if (intercepting) return;
    const input = getPromptInput();
    if (!input) return;
    const text = readInputValue(input);
    if (!text) return;
    intercepting = true;

    try {
      if (source && source.preventDefault) source.preventDefault();
      if (source && source.stopImmediatePropagation) source.stopImmediatePropagation();

      const sendBtn = getSendButton();
      if (sendBtn) sendBtn.disabled = true;
      if (input.tagName === 'TEXTAREA') input.readOnly = true;

      showLoading();
      const resp = await anonymize(text);

      if (!resp || !resp.ok) {
        const err = (resp && resp.error) ? String(resp.error) : 'Unknown error';
        showResult(text, `Anonymize failed: ${err}`);
      } else {
        const data = resp.data || {};
        const finalText = data.final_text || data.finalText || data.text || '';
        
        // Extract sensitivity info
        let sensitivityInfo = null;
        if (data.context && data.context.analysis) {
          sensitivityInfo = data.context.analysis;
        }
        
        showResult(finalText || text, null, sensitivityInfo);
      }

      const overlay = document.getElementById('prompt-anonymizer-overlay');
      const acceptBtn = overlay?.querySelector('#prompt-anonymizer-accept');
      const cancelBtn = overlay?.querySelector('#prompt-anonymizer-cancel');
      const finalTA = overlay?.querySelector('#prompt-anonymizer-final-text');

      const cleanup = () => {
        if (sendBtn) sendBtn.disabled = false;
        if (input.tagName === 'TEXTAREA') input.readOnly = false;
        hideOverlay();
        intercepting = false;
      };

      acceptBtn?.addEventListener('click', () => {
        const value = finalTA && 'value' in finalTA ? finalTA.value : '';
        setInputValue(input, value);
        lastAcceptedValue = value;
        skipNextIntercept = true; // allow next submit to go through
        cleanup();
      }, { once: true });

      cancelBtn?.addEventListener('click', () => {
        // Keep original text; allow next submit
        skipNextIntercept = true;
        cleanup();
      }, { once: true });

    } catch (e) {
      console.error('Prompt Anonymizer extension error:', e);
      showResult('', `Unexpected error: ${String(e)}`);
      const overlay = document.getElementById('prompt-anonymizer-overlay');
      const cancelBtn = overlay?.querySelector('#prompt-anonymizer-cancel');
      cancelBtn?.addEventListener('click', () => hideOverlay(), { once: true });
      intercepting = false;
    }
  }

  function attachInterceptors() {
    const input = getPromptInput();
    const btn = getSendButton();

    if (input && !input.hasAttribute('data-prompt-anonymizer-wired')) {
      input.setAttribute('data-prompt-anonymizer-wired', '1');
      input.addEventListener('keydown', (e) => {
        // Intercept Enter without Shift to submit
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
          if (skipNextIntercept) {
            // Allow this one to pass through to ChatGPT
            skipNextIntercept = false;
            return;
          }
          e.preventDefault();
          e.stopImmediatePropagation();
          handleInterceptSubmit(e);
        }
      }, true);
      // If user edits after Accept/Cancel, re-enable interception
      input.addEventListener('input', () => {
        const current = readInputValue(input);
        if (current !== lastAcceptedValue) {
          skipNextIntercept = false;
        }
      });
    }

    if (btn && !btn.hasAttribute('data-prompt-anonymizer-wired')) {
      btn.setAttribute('data-prompt-anonymizer-wired', '1');
      btn.addEventListener('click', (e) => {
        if (skipNextIntercept) {
          // Allow this one click to proceed natively
          skipNextIntercept = false;
          return;
        }
        e.preventDefault();
        e.stopImmediatePropagation();
        handleInterceptSubmit(e);
      }, true);
    }
  }

  // Observe DOM mutations to re-attach as ChatGPT UI updates
  const observer = new MutationObserver(debounce(attachInterceptors, 200));
  observer.observe(document.documentElement, { childList: true, subtree: true });

  // Initial attach
  attachInterceptors();
})();
