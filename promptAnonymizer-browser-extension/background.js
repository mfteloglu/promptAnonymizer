// Background service worker: performs cross-origin fetch to local API

// Use long-lived connection for anonymization to avoid message channel timeouts
chrome.runtime.onConnect.addListener((port) => {
  if (port.name === "prompt_anonymizer_anonymize") {
    port.onMessage.addListener((message) => {
      if (message && message.type === 'ANONYMIZE_TEXT') {
        const url = message.url || 'http://127.0.0.1:8000/anonymize_text';
        const text = message.text || '';
        
        (async () => {
          try {
            // Retrieve settings
            const settings = await new Promise((resolve) => {
              chrome.storage.sync.get(
                { 
                  useLlmAgentForPii: true, 
                  anonymizationMethod: 'pseudonymization', 
                  useReidentification: false,
                  ollamaPort: '11434',
                  ollamaModel: 'llama3.2'
                },
                (items) => resolve(items)
              );
            });

            const res = await fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                text,
                use_llm_agent_for_pii: settings.useLlmAgentForPii,
                anonymization_method: settings.anonymizationMethod,
                use_reidentification: settings.useReidentification,
                ollama_port: parseInt(settings.ollamaPort, 10) || 11434,
                model: settings.ollamaModel
              })
            });

            if (!res.ok) {
              const errTxt = await res.text().catch(() => String(res.status));
              port.postMessage({ ok: false, error: `API ${res.status}: ${errTxt}` });
            } else {
              const data = await res.json();
              port.postMessage({ ok: true, data });
            }
          } catch (e) {
            port.postMessage({ ok: false, error: String(e) });
          }
        })();
      }
    });
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === 'FETCH_MODELS') {
    const port = message.port || '11434';
    (async () => {
      try {
        const res = await fetch(`http://localhost:${port}/api/tags`);
        if (!res.ok) {
          sendResponse({ ok: false, error: `Ollama ${res.status}` });
          return;
        }
        const data = await res.json();
        sendResponse({ ok: true, data });
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    })();
    return true;
  }
});
