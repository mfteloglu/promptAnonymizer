// Saves options to chrome.storage
const saveOptions = () => {
  const useLlmAgentForPii = document.getElementById('use_llm_agent_for_pii').checked;
  const anonymizationMethod = document.getElementById('anonymization_method').value;
  const useReidentification = document.getElementById('use_reidentification').checked;
  const ollamaPort = document.getElementById('ollama_port').value;
  const ollamaModel = document.getElementById('ollama_model').value;

  chrome.storage.sync.set(
    { 
      useLlmAgentForPii, 
      anonymizationMethod, 
      useReidentification,
      ollamaPort,
      ollamaModel
    },
    () => {
      // Update status to let user know options were saved.
      const status = document.getElementById('status');
      status.textContent = 'Options saved.';
      status.className = 'success';
      setTimeout(() => {
        status.textContent = '';
        status.className = '';
      }, 1500);
    }
  );
};

// Restores select box and checkbox state using the preferences
// stored in chrome.storage.
const restoreOptions = () => {
  chrome.storage.sync.get(
    { 
      useLlmAgentForPii: true, 
      anonymizationMethod: 'pseudonymization',
      useReidentification: false,
      ollamaPort: '11434',
      ollamaModel: 'llama3.2'
    },
    (items) => {
      document.getElementById('use_llm_agent_for_pii').checked = items.useLlmAgentForPii;
      document.getElementById('anonymization_method').value = items.anonymizationMethod;
      document.getElementById('use_reidentification').checked = items.useReidentification;
      document.getElementById('ollama_port').value = items.ollamaPort;
      
      // Populate model dropdown if saved, otherwise keep default
      const modelSelect = document.getElementById('ollama_model');
      
      // If we have a saved model but it's not in the list (which is just llama3.2 initially), add it
      if (items.ollamaModel && items.ollamaModel !== 'llama3.2') {
        let exists = false;
        for (let i = 0; i < modelSelect.options.length; i++) {
          if (modelSelect.options[i].value === items.ollamaModel) {
            exists = true;
            break;
          }
        }
        if (!exists) {
          const option = document.createElement('option');
          option.value = items.ollamaModel;
          option.text = items.ollamaModel;
          modelSelect.add(option);
        }
      }
      modelSelect.value = items.ollamaModel;
    }
  );
};

const fetchModels = async () => {
  const port = document.getElementById('ollama_port').value || '11434';
  const status = document.getElementById('status');
  const btn = document.getElementById('fetch_models');
  const select = document.getElementById('ollama_model');
  
  status.textContent = 'Fetching models...';
  status.className = '';
  btn.disabled = true;
  
  try {
    // Use background script to fetch to avoid CORS
    const response = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: 'FETCH_MODELS', port }, (response) => {
        resolve(response);
      });
    });

    if (!response || !response.ok) {
      throw new Error(response ? response.error : 'Unknown error');
    }
    
    const data = response.data;
    const models = data.models || [];
    
    const currentSelection = select.value;
    select.innerHTML = '';
    
    if (models.length === 0) {
      const option = document.createElement('option');
      option.value = 'llama3.2';
      option.text = 'llama3.2 (Default)';
      select.add(option);
    } else {
      models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.name;
        option.text = model.name;
        select.add(option);
      });
    }
    
    // Restore selection if possible
    let restored = false;
    if (currentSelection) {
      for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].value === currentSelection) {
          select.value = currentSelection;
          restored = true;
          break;
        }
      }
    }
    
    // If current selection is not in the new list, select the first one
    if (!restored && select.options.length > 0) {
        select.selectedIndex = 0;
        saveOptions(); // Save the new valid selection
    }
    
    status.textContent = `Found ${models.length} models.`;
    status.className = 'success';
    
  } catch (error) {
    console.error(error);
    status.textContent = `Error: ${error.message}`;
    status.className = 'error';
    
    // Ensure at least one option exists
    if (select.options.length === 0) {
      const option = document.createElement('option');
      option.value = 'llama3.2';
      option.text = 'llama3.2';
      select.add(option);
    }
  } finally {
    btn.disabled = false;
    setTimeout(() => {
      if (status.className === 'success') status.textContent = '';
    }, 2000);
  }
};

document.addEventListener('DOMContentLoaded', restoreOptions);
document.getElementById('use_llm_agent_for_pii').addEventListener('change', saveOptions);
document.getElementById('anonymization_method').addEventListener('change', saveOptions);
document.getElementById('use_reidentification').addEventListener('change', saveOptions);
document.getElementById('ollama_port').addEventListener('change', saveOptions);
document.getElementById('ollama_model').addEventListener('change', saveOptions);
document.getElementById('fetch_models').addEventListener('click', fetchModels);
