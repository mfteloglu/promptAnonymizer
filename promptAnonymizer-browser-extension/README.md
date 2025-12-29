# Prompt Anonymizer Chrome Extension

Intercepts prompts on chatgpt.com, calls your local API (`/anonymize_text`), shows the anonymized text, and lets you accept to replace the prompt.

## Install (Load Unpacked)

1. Build nothing — this is a plain MV3 extension.
2. Visit `chrome://extensions`.
3. Enable Developer mode (top right).
4. Click "Load unpacked" and select this folder: `promptAnonymizer-browser-extension`.

## Permissions

- `host_permissions`: `http://127.0.0.1:8000/*`, `http://localhost:8000/*` for the local API.

## Usage

1. Start your FastAPI server locally at `http://127.0.0.1:8000`.
2. Open `https://chatgpt.com/`.
3. Type a prompt and press Enter or click Send.
4. The extension intercepts, shows a loading screen, calls the anonymizer, then displays the `final_text`.
5. If the content is flagged as sensitive (e.g., MEDICAL, FINANCIAL), a warning will be displayed above the text.
6. Click "Accept" to replace the prompt box contents with the anonymized text. You can then send it.

## Notes

- If ChatGPT’s UI changes, the extension uses a MutationObserver to rewire listeners.
- If the API returns a different property, it tries `final_text`, then `finalText`, then `text`.
- For development, ensure your API is reachable and not blocked by security tools.
