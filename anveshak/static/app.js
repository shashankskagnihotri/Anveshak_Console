let sessionId = null;
let activeRunId = null;
let activeUserMessageWrapper = null;
let activeAssistantNode = null;
let activeReasoningNode = null;
let activeActivityFeed = null;
let activeActivityLastText = "";
let activeRunPhase = null;
let promptLocked = false;
let runtimeReady = false;
let runtimeStream = null;
let runtimeReconnectTimer = null;
let editingApiCallId = null;
let currentWebMode = "auto";
let currentApiWebMode = "auto";
let currentThemePreference = "system";
let apiUseUserContext = false;
let apiInstanceMode = "independent";
let runtimeStatusSnapshot = null;
let pendingApiDeleteId = null;
let apiCallsCache = [];
let lastHuggingFaceAuthPromptVersion = -1;
const pendingFiles = [];

const API_DOCS_LOCAL_PATH = "./Documentations/API_CALLS.md";
const API_DOCS_GITHUB_URL = "https://github.com/shashankskagnihotri/Anveshak_Console/blob/main/Documentations/API_CALLS.md";
const THEME_STORAGE_KEY = "anveshak-theme-preference";
const systemThemeQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

const runtimeOverlayEl = document.getElementById("runtime-overlay");
const runtimeMessageEl = document.getElementById("runtime-message");
const runtimeProgressBarEl = document.getElementById("runtime-progress-bar");
const runtimePercentEl = document.getElementById("runtime-percent");
const runtimeFileEl = document.getElementById("runtime-file");
const runtimeModelEl = document.getElementById("runtime-model");
const runtimeEmbeddingEl = document.getElementById("runtime-embedding");
const runtimeSummaryEl = document.getElementById("runtime-summary");

const chatViewEl = document.getElementById("chat-view");
const apiViewEl = document.getElementById("api-view");
const themeToggleEl = document.getElementById("theme-toggle");
const themeOptionEls = Array.from(themeToggleEl?.querySelectorAll(".theme-option") || []);
const homeButtonEl = document.getElementById("home-button");
const newApiCallEl = document.getElementById("new-api-call");
const existingApiCallsEl = document.getElementById("existing-api-calls");

const messagesEl = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const fileInputEl = document.getElementById("file-input");
const fileListEl = document.getElementById("file-list");
const sendButtonEl = document.getElementById("send-button");
const webModeToggleEl = document.getElementById("web-mode-toggle");
const webModeOptionEls = Array.from(document.querySelectorAll(".web-mode-option"));
const steerInputEl = document.getElementById("steer-input");
const steerButtonEl = document.getElementById("steer-button");
const steerBadgeEl = document.getElementById("steer-badge");
const steerHintEl = document.getElementById("steer-hint");
const dropZoneEl = document.getElementById("drop-zone");
let steerHintResetTimer = null;

const apiBuilderTabEl = document.getElementById("api-builder-tab");
const apiKeysTabEl = document.getElementById("api-keys-tab");
const apiBuilderScreenEl = document.getElementById("api-builder-screen");
const apiKeysScreenEl = document.getElementById("api-keys-screen");
const apiCreateFromKeysEl = document.getElementById("api-create-from-keys");
const apiCallListEl = document.getElementById("api-call-list");
const apiKeySummaryEl = document.getElementById("api-key-summary");
const apiFormTitleEl = document.getElementById("api-form-title");
const apiFormBadgeEl = document.getElementById("api-form-badge");
const apiFormStatusEl = document.getElementById("api-form-status");
const apiKeyBoxEl = document.getElementById("api-key-box");
const apiKeyValueEl = document.getElementById("api-key-value");
const apiCopyKeyEl = document.getElementById("api-copy-key");
const apiOpenKeyDocsEl = document.getElementById("api-open-key-docs");
const apiNameEl = document.getElementById("api-name");
const apiModelEl = document.getElementById("api-model");
const apiEmbeddingModelEl = document.getElementById("api-embedding-model");
const apiSystemPromptEl = document.getElementById("api-system-prompt");
const apiInputTemplateEl = document.getElementById("api-input-template");
const apiResponseInstructionsEl = document.getElementById("api-response-instructions");
const apiResponseModeEl = document.getElementById("api-response-mode");
const apiWebModeToggleEl = document.getElementById("api-web-mode-toggle");
const apiWebModeOptionEls = Array.from(apiWebModeToggleEl?.querySelectorAll(".web-mode-option") || []);
const apiUserContextToggleEl = document.getElementById("api-user-context-toggle");
const apiUserContextOptionEls = Array.from(apiUserContextToggleEl?.querySelectorAll(".segmented-option") || []);
const apiInstanceModeToggleEl = document.getElementById("api-instance-mode-toggle");
const apiInstanceModeOptionEls = Array.from(apiInstanceModeToggleEl?.querySelectorAll(".segmented-option") || []);
const apiCurrentModelEl = document.getElementById("api-current-model");
const apiCurrentEmbeddingEl = document.getElementById("api-current-embedding");
const apiDocsPathEl = document.getElementById("api-docs-path");
const apiDocsLinkEl = document.getElementById("api-docs-link");
const apiInvokeEndpointEl = document.getElementById("api-invoke-endpoint");
const apiCurlPreviewEl = document.getElementById("api-curl-preview");
const saveApiCallEl = document.getElementById("save-api-call");
const resetApiFormEl = document.getElementById("reset-api-form");
const apiKeyModalEl = document.getElementById("api-key-modal");
const apiKeyModalValueEl = document.getElementById("api-key-modal-value");
const apiKeyModalCopyEl = document.getElementById("api-key-modal-copy");
const apiKeyModalDocsLinkEl = document.getElementById("api-key-modal-docs-link");
const apiKeyModalDocsPathEl = document.getElementById("api-key-modal-docs-path");
const apiKeyModalCurlEl = document.getElementById("api-key-modal-curl");
const apiDeleteModalEl = document.getElementById("api-delete-modal");
const apiDeletePreviewEl = document.getElementById("api-delete-preview");
const apiDeleteConfirmEl = document.getElementById("api-delete-confirm");
const huggingFaceTokenModalEl = document.getElementById("huggingface-token-modal");
const huggingFaceTokenMessageEl = document.getElementById("huggingface-token-message");
const huggingFaceTokenModelEl = document.getElementById("huggingface-token-model");
const huggingFaceTokenEnvVarEl = document.getElementById("huggingface-token-env-var");
const huggingFaceTokenExportCommandEl = document.getElementById("huggingface-token-export-command");
const huggingFaceTokenInputEl = document.getElementById("huggingface-token-input");
const huggingFaceTokenStatusEl = document.getElementById("huggingface-token-status");
const huggingFaceTokenGuideLinkEl = document.getElementById("huggingface-token-guide-link");
const huggingFaceTokenSettingsLinkEl = document.getElementById("huggingface-token-settings-link");
const huggingFaceTokenSubmitEl = document.getElementById("huggingface-token-submit");
const modalCloseEls = Array.from(document.querySelectorAll("[data-close-modal]"));

async function ensureSession() {
  if (sessionId) return sessionId;
  const response = await fetch("/api/sessions", { method: "POST" });
  const payload = await response.json();
  sessionId = payload.session_id;
  return sessionId;
}

function normalizeWebMode(mode) {
  if (mode === "off" || mode === "always") return mode;
  return "auto";
}

function normalizeThemePreference(mode) {
  if (mode === "light" || mode === "night") return mode;
  return "system";
}

function resolveTheme(preference) {
  const normalizedPreference = normalizeThemePreference(preference);
  if (normalizedPreference !== "system") return normalizedPreference;
  return systemThemeQuery?.matches ? "night" : "light";
}

function persistThemePreference(preference) {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, preference);
  } catch (error) {
    // Ignore storage failures and keep the active in-memory preference.
  }
}

function loadThemePreference() {
  try {
    return normalizeThemePreference(window.localStorage.getItem(THEME_STORAGE_KEY));
  } catch (error) {
    return "system";
  }
}

function applyThemePreference(preference, persist = true) {
  currentThemePreference = normalizeThemePreference(preference);
  const resolvedTheme = resolveTheme(currentThemePreference);

  document.documentElement.dataset.themePreference = currentThemePreference;
  document.documentElement.dataset.theme = resolvedTheme;

  if (themeToggleEl) {
    themeToggleEl.classList.remove("mode-light", "mode-system", "mode-night");
    themeToggleEl.classList.add(`mode-${currentThemePreference}`);
  }

  themeOptionEls.forEach((option) => {
    const selected = option.dataset.themeMode === currentThemePreference;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-checked", String(selected));
  });

  if (persist) {
    persistThemePreference(currentThemePreference);
  }
}

function applyTriModeToggle(containerEl, optionEls, mode) {
  if (!containerEl) return;
  containerEl.classList.remove("mode-off", "mode-auto", "mode-always");
  containerEl.classList.add(`mode-${mode}`);
  optionEls.forEach((option) => {
    const selected = option.dataset.webMode === mode;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-checked", String(selected));
  });
}

function applyWebMode(mode) {
  currentWebMode = normalizeWebMode(mode);
  applyTriModeToggle(webModeToggleEl, webModeOptionEls, currentWebMode);
}

function applyApiWebMode(mode) {
  currentApiWebMode = normalizeWebMode(mode);
  applyTriModeToggle(apiWebModeToggleEl, apiWebModeOptionEls, currentApiWebMode);
  updateApiUsagePreview();
}

function applySegmentedToggle(containerEl, optionEls, isOn, valueAttribute) {
  if (!containerEl) return;
  containerEl.classList.remove("mode-off", "mode-on");
  containerEl.classList.add(isOn ? "mode-on" : "mode-off");
  optionEls.forEach((option) => {
    const selected = valueAttribute ? option.dataset[valueAttribute] === isOn : false;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-checked", String(selected));
  });
}

function applyUserContextToggle(enabled) {
  apiUseUserContext = Boolean(enabled);
  if (!apiUserContextToggleEl) return;
  apiUserContextToggleEl.classList.toggle("mode-on", apiUseUserContext);
  apiUserContextToggleEl.classList.toggle("mode-off", !apiUseUserContext);
  apiUserContextOptionEls.forEach((option) => {
    const selected = (option.dataset.booleanMode === "on") === apiUseUserContext;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-checked", String(selected));
  });
}

function applyInstanceMode(mode) {
  apiInstanceMode = mode === "remember" ? "remember" : "independent";
  if (!apiInstanceModeToggleEl) return;
  apiInstanceModeToggleEl.classList.toggle("mode-on", apiInstanceMode === "remember");
  apiInstanceModeToggleEl.classList.toggle("mode-off", apiInstanceMode !== "remember");
  apiInstanceModeOptionEls.forEach((option) => {
    const selected = option.dataset.instanceMode === apiInstanceMode;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-checked", String(selected));
  });
}

function showView(view) {
  chatViewEl.classList.toggle("active", view === "chat");
  apiViewEl.classList.toggle("active", view === "api");
}

function showApiScreen(screen) {
  const builder = screen !== "keys";
  apiBuilderScreenEl.classList.toggle("active", builder);
  apiKeysScreenEl.classList.toggle("active", !builder);
  apiBuilderTabEl.classList.toggle("active", builder);
  apiKeysTabEl.classList.toggle("active", !builder);
}

function renderFileChips() {
  fileListEl.innerHTML = "";
  pendingFiles.forEach((file, index) => {
    const chip = document.createElement("span");
    chip.className = "file-chip";
    chip.textContent = file.name;
    chip.onclick = () => {
      pendingFiles.splice(index, 1);
      renderFileChips();
    };
    fileListEl.appendChild(chip);
  });
}

function describeAttachment(file) {
  const parts = file.name.split(".");
  const extension = parts.length > 1 ? parts.pop().toUpperCase() : "FILE";
  const mimeType = String(file.type || "");
  const isImage = mimeType.startsWith("image/");
  return {
    name: file.name,
    badge: extension || "FILE",
    kind: isImage ? "image" : "file",
    previewUrl: isImage ? URL.createObjectURL(file) : null,
  };
}

function appendAttachmentPreviews(wrapper, attachments) {
  if (!attachments || attachments.length === 0) return;

  const container = document.createElement("div");
  container.className = "message-attachments";

  attachments.forEach((attachment) => {
    const card = document.createElement("div");
    card.className = `attachment-card ${attachment.kind}`;
    card.title = attachment.name;

    if (attachment.kind === "image" && attachment.previewUrl) {
      const image = document.createElement("img");
      image.className = "attachment-thumb";
      image.alt = attachment.name;
      image.src = attachment.previewUrl;
      image.addEventListener("load", () => URL.revokeObjectURL(attachment.previewUrl), { once: true });
      image.addEventListener("error", () => URL.revokeObjectURL(attachment.previewUrl), { once: true });
      card.appendChild(image);
    }

    const meta = document.createElement("div");
    meta.className = "attachment-meta";

    const badge = document.createElement("span");
    badge.className = "attachment-badge";
    badge.textContent = attachment.badge;

    const name = document.createElement("span");
    name.className = "attachment-name";
    name.textContent = attachment.name;

    meta.append(badge, name);
    card.appendChild(meta);
    container.appendChild(card);
  });

  wrapper.appendChild(container);
}

function addMessage(role, content, options = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "label";
  label.textContent = role === "user" ? "You" : "Anveshak";

  const activity = document.createElement("div");
  activity.className = "activity-feed hidden";

  const body = document.createElement("div");
  body.className = "content";
  body.textContent = content;

  wrapper.append(label);
  if (role === "assistant") {
    wrapper.append(activity);
  }
  wrapper.append(body);
  appendAttachmentPreviews(wrapper, options.attachments || []);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { wrapper, body, activity };
}

function appendSteeringNote(messageWrapper, text) {
  if (!messageWrapper || !text) return;

  let container = messageWrapper.querySelector(".steering-notes");
  if (!container) {
    container = document.createElement("div");
    container.className = "steering-notes";
    messageWrapper.appendChild(container);
  }

  const note = document.createElement("div");
  note.className = "steering-note";

  const tag = document.createElement("div");
  tag.className = "steering-note-tag";
  tag.textContent = "Steering note";

  const body = document.createElement("div");
  body.className = "steering-note-text";
  body.textContent = text;

  note.append(tag, body);
  container.appendChild(note);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updateRunControls() {
  const canSendPrompt = runtimeReady && !promptLocked;
  const canSteer = Boolean(activeRunId) && activeRunPhase === "generation";

  sendButtonEl.disabled = !canSendPrompt;
  promptEl.disabled = false;
  fileInputEl.disabled = !canSendPrompt;

  steerInputEl.disabled = false;
  steerButtonEl.disabled = false;
  steerButtonEl.classList.toggle("is-inactive", !canSteer);
  steerButtonEl.setAttribute("aria-disabled", String(!canSteer));

  if (canSteer) {
    if (steerHintEl) {
      steerHintEl.classList.remove("warning-flash");
    }
    if (steerHintResetTimer) {
      window.clearTimeout(steerHintResetTimer);
      steerHintResetTimer = null;
    }
    steerBadgeEl.textContent = "Active";
    steerBadgeEl.className = "badge active";
    return;
  }
  if (activeRunId) {
    steerBadgeEl.textContent = "Waiting";
    steerBadgeEl.className = "badge muted";
    return;
  }
  steerBadgeEl.textContent = "Idle";
  steerBadgeEl.className = "badge muted";
}

function flashSteerHintWarning() {
  if (!steerHintEl) return;
  steerHintEl.classList.add("warning-flash");
  if (steerHintResetTimer) {
    window.clearTimeout(steerHintResetTimer);
  }
  steerHintResetTimer = window.setTimeout(() => {
    steerHintEl.classList.remove("warning-flash");
    steerHintResetTimer = null;
  }, 5000);
}

function pushActivity(text, phase) {
  if (!activeActivityFeed || !text || text === activeActivityLastText) return;
  activeActivityLastText = text;
  activeActivityFeed.classList.remove("hidden");

  Array.from(activeActivityFeed.children).forEach((row, index) => {
    row.classList.remove("live");
    row.classList.toggle("stale", index >= 0);
  });

  const row = document.createElement("div");
  row.className = "activity-row live";
  row.dataset.phase = phase || "working";

  const spinner = document.createElement("span");
  spinner.className = "activity-spinner";

  const label = document.createElement("div");
  label.className = "activity-label";
  label.textContent = text;

  row.append(spinner, label);
  activeActivityFeed.prepend(row);

  while (activeActivityFeed.children.length > 5) {
    activeActivityFeed.removeChild(activeActivityFeed.lastElementChild);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function settleActivity(text) {
  if (!activeActivityFeed) return;
  const latest = activeActivityFeed.firstElementChild;
  if (latest) {
    latest.classList.remove("live");
    latest.classList.add("completed");
    const label = latest.querySelector(".activity-label");
    if (label && text) label.textContent = text;
  }
}

async function sendPrompt() {
  if (!runtimeReady || promptLocked) return;
  const text = promptEl.value.trim();
  if (!text && pendingFiles.length === 0) return;
  await ensureSession();
  promptLocked = true;
  activeRunPhase = "submit";
  updateRunControls();

  const messageAttachments = pendingFiles.map((file) => describeAttachment(file));
  const userMessage = addMessage("user", text || "[Attachment only]", { attachments: messageAttachments });
  activeUserMessageWrapper = userMessage.wrapper;
  const assistant = addMessage("assistant", "");
  activeAssistantNode = assistant.body;
  activeActivityFeed = assistant.activity;
  activeActivityLastText = "";

  const reasoning = document.createElement("div");
  reasoning.className = "reasoning";
  assistant.wrapper.appendChild(reasoning);
  activeReasoningNode = reasoning;

  const form = new FormData();
  form.append("text", text);
  form.append("web_mode", currentWebMode);
  pendingFiles.forEach((file) => form.append("files", file));

  promptEl.value = "";
  pendingFiles.length = 0;
  renderFileChips();

  pushActivity("Submitting prompt to Anveshak Console", "submit");

  try {
    const response = await fetch(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "The backend rejected the prompt submission.");
    }
    const payload = await response.json();
    activeRunId = payload.run_id;
    updateRunControls();
    streamRun(activeRunId, assistant.wrapper);
  } catch (error) {
    addErrorMessage(error.message || "The prompt could not be submitted.");
    settleActivity("Submission failed");
    cleanupRun(null);
  }
}

function streamRun(runId, messageWrapper) {
  const source = new EventSource(`/api/runs/${runId}/events`);
  let finished = false;

  function finishRun(statusText) {
    if (finished) return;
    finished = true;
    settleActivity(statusText);
    cleanupRun(source);
  }

  source.addEventListener("status", (event) => {
    const payload = JSON.parse(event.data).payload;
    activeRunPhase = payload.phase || activeRunPhase;
    updateRunControls();
    pushActivity(payload.text, payload.phase);
  });
  source.addEventListener("reasoning", (event) => {
    const payload = JSON.parse(event.data).payload;
    activeReasoningNode.textContent += payload.text;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
  source.addEventListener("warning", (event) => {
    const payload = JSON.parse(event.data).payload;
    addWarningMessage(payload.text);
  });
  source.addEventListener("token", (event) => {
    const payload = JSON.parse(event.data).payload;
    activeAssistantNode.textContent += payload.text;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
  source.addEventListener("done", (event) => {
    const payload = JSON.parse(event.data).payload;
    appendSourceChips(messageWrapper, payload.citations || []);
    finishRun("Answer complete");
  });
  source.addEventListener("error", (event) => {
    if (event.data) {
      const payload = JSON.parse(event.data).payload;
      addErrorMessage(payload.message || "The current run failed.");
      finishRun("Run failed");
      return;
    }
    if (!finished) {
      addErrorMessage("The browser lost contact with the run. Restart the web server if it was interrupted.");
      finishRun("Connection lost");
    }
  });
}

function addWarningMessage(text) {
  const warning = addMessage("warning", text);
  warning.wrapper.querySelector(".label").textContent = "Compatibility warning";
}

function addErrorMessage(text) {
  const error = addMessage("error", text);
  error.wrapper.querySelector(".label").textContent = "Run error";
}

function appendSourceChips(messageWrapper, citations) {
  if (citations.length === 0) return;
  const sources = document.createElement("div");
  sources.className = "sources";
  citations.slice(0, 12).forEach((citation) => {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = citation.label || citation.source_id;
    chip.title = citation.metadata?.source_path || citation.metadata?.url || "";
    sources.appendChild(chip);
  });
  messageWrapper.appendChild(sources);
}

function cleanupRun(source) {
  if (source) source.close();
  activeRunId = null;
  activeRunPhase = null;
  activeUserMessageWrapper = null;
  promptLocked = false;
  activeAssistantNode = null;
  activeReasoningNode = null;
  activeActivityFeed = null;
  activeActivityLastText = "";
  updateRunControls();
}

async function sendSteeringNote() {
  const text = steerInputEl.value.trim();
  if (!activeRunId || activeRunPhase !== "generation") {
    flashSteerHintWarning();
    return;
  }
  if (!text) return;
  const form = new FormData();
  form.append("text", text);
  const response = await fetch(`/api/runs/${activeRunId}/steer`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    addWarningMessage(payload.detail || "Steering is only available while the model is actively generating an answer.");
    return;
  }
  appendSteeringNote(activeUserMessageWrapper, text);
  steerInputEl.value = "";
}

async function refreshRuntimeStatus() {
  const response = await fetch("/api/runtime/status");
  const payload = await response.json();
  applyRuntimeStatus(payload);
}

function syncApiRuntimeSummary(payload) {
  runtimeStatusSnapshot = payload;
  if (apiCurrentModelEl) apiCurrentModelEl.textContent = payload.model_id;
  if (apiCurrentEmbeddingEl) apiCurrentEmbeddingEl.textContent = payload.embedding_model_id;
  if (apiDocsPathEl) apiDocsPathEl.textContent = API_DOCS_LOCAL_PATH;
  if (apiDocsLinkEl) apiDocsLinkEl.href = API_DOCS_GITHUB_URL;
  if (apiKeyModalDocsLinkEl) apiKeyModalDocsLinkEl.href = API_DOCS_GITHUB_URL;
  if (apiKeyModalDocsPathEl) apiKeyModalDocsPathEl.textContent = API_DOCS_LOCAL_PATH;

  if (!editingApiCallId) {
    apiModelEl.value = payload.model_id;
    apiEmbeddingModelEl.value = payload.embedding_model_id;
  }
  updateApiUsagePreview();
}

function applyRuntimeStatus(payload) {
  runtimeMessageEl.textContent = payload.message;
  runtimeProgressBarEl.style.width = `${Math.round((payload.progress || 0) * 100)}%`;
  runtimePercentEl.textContent = `${Math.round((payload.progress || 0) * 100)}%`;
  runtimeFileEl.textContent = payload.current_file || "Waiting...";
  runtimeModelEl.textContent = payload.model_id;
  runtimeEmbeddingEl.textContent = payload.embedding_model_id;
  renderRuntimeSummary(payload);
  syncApiRuntimeSummary(payload);
  maybeShowHuggingFaceTokenModal(payload);

  runtimeReady = Boolean(payload.ready);
  updateRunControls();

  runtimeOverlayEl.classList.toggle("ready", runtimeReady);
  if (!runtimeReady) {
    runtimeOverlayEl.classList.remove("ready");
  }
}

function connectRuntimeStream() {
  if (runtimeStream) {
    runtimeStream.close();
  }

  runtimeStream = new EventSource("/api/runtime/events");
  runtimeStream.addEventListener("status", (event) => {
    const payload = JSON.parse(event.data);
    applyRuntimeStatus(payload);
  });
  runtimeStream.onerror = () => {
    if (runtimeStream) {
      runtimeStream.close();
      runtimeStream = null;
    }
    if (runtimeReconnectTimer) {
      window.clearTimeout(runtimeReconnectTimer);
    }
    runtimeReconnectTimer = window.setTimeout(() => {
      connectRuntimeStream();
    }, 1500);
  };
}

function renderRuntimeSummary(status) {
  runtimeSummaryEl.innerHTML = "";
  [
    ["Phase", status.phase],
    ["Reasoning model", status.model_id],
    ["Embedding model for RAGs", status.embedding_model_id],
    ["Checkpoints", status.checkpoints_dir],
  ].forEach(([label, value]) => {
    const node = document.createElement("div");
    node.className = "runtime-item";
    const key = document.createElement("span");
    key.className = "runtime-key";
    key.textContent = label;
    const detail = document.createElement("div");
    detail.className = "runtime-value";
    detail.textContent = value;
    detail.title = value;
    node.append(key, detail);
    runtimeSummaryEl.appendChild(node);
  });
}

async function loadApiCalls() {
  const response = await fetch("/api/api-calls");
  const payload = await response.json();
  apiCallsCache = payload.items || [];
  renderApiCallList();
}

function maskApiKey(apiKey) {
  if (!apiKey) return "";
  if (apiKey.length <= 14) return apiKey;
  return `${apiKey.slice(0, 8)}...${apiKey.slice(-4)}`;
}

function setApiFormStatus(message, tone = "neutral") {
  if (!apiFormStatusEl) return;
  apiFormStatusEl.textContent = message;
  apiFormStatusEl.classList.remove("success-text", "error-text");
  if (tone === "success") apiFormStatusEl.classList.add("success-text");
  if (tone === "error") apiFormStatusEl.classList.add("error-text");
}

function setApiFormBadge(label, active = false) {
  if (!apiFormBadgeEl) return;
  apiFormBadgeEl.textContent = label;
  apiFormBadgeEl.className = active ? "badge active" : "badge muted";
}

function renderApiKeyBox(payload) {
  if (!payload || !payload.api_key) {
    apiKeyBoxEl.classList.add("hidden");
    apiKeyValueEl.textContent = "";
    updateApiUsagePreview();
    return;
  }
  apiKeyBoxEl.classList.remove("hidden");
  apiKeyValueEl.textContent = payload.api_key;
  updateApiUsagePreview(payload);
}

function updateApiUsagePreview(payload = null) {
  const callRef = payload?.call_id || editingApiCallId || "<call_id>";
  const endpoint = `${window.location.origin}/v1/api-calls/${callRef}/invoke`;
  const keyPlaceholder = payload?.api_key || "<API_KEY>";
  if (apiInvokeEndpointEl) {
    apiInvokeEndpointEl.textContent = `POST ${endpoint}`;
  }
  if (apiCurlPreviewEl) {
    apiCurlPreviewEl.textContent = [
      `curl -X POST ${endpoint} \\`,
      `  -H "Authorization: Bearer ${payload?.api_key ? "<paste copied key here>" : keyPlaceholder}" \\`,
      '  -H "Content-Type: application/json" \\',
      '  -d \'{',
      '    "input": "Summarize the attached evidence.",',
      '    "variables": {"paper_id": 42}',
      "  }'",
    ].join("\n");
  }
}

function openModal(modalEl) {
  if (!modalEl) return;
  modalEl.classList.remove("hidden");
}

function closeModal(modalEl) {
  if (!modalEl) return;
  modalEl.classList.add("hidden");
}

function setHuggingFaceTokenStatus(message, tone = "neutral") {
  if (!huggingFaceTokenStatusEl) return;
  huggingFaceTokenStatusEl.textContent = message;
  huggingFaceTokenStatusEl.classList.remove("success-text", "error-text");
  if (tone === "success") huggingFaceTokenStatusEl.classList.add("success-text");
  if (tone === "error") huggingFaceTokenStatusEl.classList.add("error-text");
}

function populateHuggingFaceTokenModal(payload) {
  const envVar = payload.huggingface_token_env_var || "HUGGINGFACE_HUB_TOKEN";
  const aliasEnvVars = Array.isArray(payload.huggingface_token_alias_env_vars)
    ? payload.huggingface_token_alias_env_vars.filter(Boolean)
    : [];
  const modelId = payload.huggingface_auth_model_id || payload.model_id || "-";
  if (huggingFaceTokenMessageEl) {
    huggingFaceTokenMessageEl.textContent = payload.huggingface_auth_message
      || "This model appears to need a personal Hugging Face token before Anveshak can finish preparing the runtime.";
  }
  if (huggingFaceTokenModelEl) huggingFaceTokenModelEl.textContent = modelId;
  if (huggingFaceTokenEnvVarEl) huggingFaceTokenEnvVarEl.textContent = envVar;
  if (huggingFaceTokenExportCommandEl) huggingFaceTokenExportCommandEl.textContent = `export ${envVar}=hf_...`;
  if (huggingFaceTokenGuideLinkEl) {
    huggingFaceTokenGuideLinkEl.href = payload.huggingface_token_guide_url || "https://huggingface.co/docs/hub/main/security-tokens";
  }
  if (huggingFaceTokenSettingsLinkEl) {
    huggingFaceTokenSettingsLinkEl.href = payload.huggingface_token_settings_url || "https://huggingface.co/settings/tokens";
  }
  const aliasNote = aliasEnvVars.length ? ` It also treats ${aliasEnvVars.join(", ")} as the same token.` : "";
  setHuggingFaceTokenStatus(`Anveshak checks ${envVar} automatically on startup.${aliasNote} Most public models do not need it.`);
}

function maybeShowHuggingFaceTokenModal(payload) {
  if (!payload?.huggingface_auth_required) {
    closeModal(huggingFaceTokenModalEl);
    return;
  }

  populateHuggingFaceTokenModal(payload);

  const version = Number(payload.version || 0);
  if (!huggingFaceTokenModalEl?.classList.contains("hidden") && version === lastHuggingFaceAuthPromptVersion) {
    return;
  }

  lastHuggingFaceAuthPromptVersion = version;
  openModal(huggingFaceTokenModalEl);
  huggingFaceTokenInputEl?.focus();
  huggingFaceTokenInputEl?.select();
}

async function submitHuggingFaceToken() {
  const token = huggingFaceTokenInputEl?.value.trim() || "";
  if (!token) {
    setHuggingFaceTokenStatus("Enter a Hugging Face token before retrying.", "error");
    huggingFaceTokenInputEl?.focus();
    return;
  }

  if (huggingFaceTokenSubmitEl) huggingFaceTokenSubmitEl.disabled = true;
  if (huggingFaceTokenInputEl) huggingFaceTokenInputEl.disabled = true;
  setHuggingFaceTokenStatus("Validating the token and retrying runtime preparation...");

  try {
    const response = await fetch("/api/runtime/huggingface-token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "The Hugging Face token could not be accepted.");
    }
    if (huggingFaceTokenInputEl) huggingFaceTokenInputEl.value = "";
    setHuggingFaceTokenStatus("Token accepted. Retrying gated model access.", "success");
    closeModal(huggingFaceTokenModalEl);
    applyRuntimeStatus(payload);
  } catch (error) {
    setHuggingFaceTokenStatus(error.message || "The Hugging Face token could not be accepted.", "error");
    huggingFaceTokenInputEl?.focus();
    huggingFaceTokenInputEl?.select();
  } finally {
    if (huggingFaceTokenSubmitEl) huggingFaceTokenSubmitEl.disabled = false;
    if (huggingFaceTokenInputEl) huggingFaceTokenInputEl.disabled = false;
  }
}

async function copyTextToClipboard(text, successMessage) {
  try {
    await navigator.clipboard.writeText(text);
    if (successMessage) {
      setApiFormStatus(successMessage, "success");
    }
  } catch (error) {
    setApiFormStatus("Clipboard copy failed in this browser.", "error");
  }
}

function showApiKeyModal(payload) {
  const endpoint = `${window.location.origin}/v1/api-calls/${payload.call_id}/invoke`;
  apiKeyModalValueEl.textContent = payload.api_key;
  apiKeyModalDocsLinkEl.href = API_DOCS_GITHUB_URL;
  apiKeyModalDocsPathEl.textContent = API_DOCS_LOCAL_PATH;
  apiKeyModalCurlEl.textContent = [
    `curl -X POST ${endpoint} \\`,
    `  -H "Authorization: Bearer ${payload.api_key}" \\`,
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "input": "Run the saved workflow on this payload.",',
    '    "variables": {"example": true}',
    "  }'",
  ].join("\n");
  openModal(apiKeyModalEl);
}

function renderDeletePreview(item) {
  apiDeletePreviewEl.textContent = JSON.stringify(
    {
      name: item.name,
      call_id: item.call_id,
      model_id: item.model_id,
      embedding_model_id: item.embedding_model_id,
      system_prompt: item.system_prompt,
      web_mode: item.web_mode,
      use_user_context: item.use_user_context,
      instance_mode: item.instance_mode,
      response_mode: item.response_mode,
      input_template: item.input_template,
      response_instructions: item.response_instructions,
    },
    null,
    2,
  );
}

function apiIconSvg(kind) {
  if (kind === "edit") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 16.5V20h3.5L18 9.5 14.5 6 4 16.5zm15.7-9.3a1 1 0 0 0 0-1.4l-1.5-1.5a1 1 0 0 0-1.4 0L15.4 5.7 18.9 9l.8-.8z"/></svg>';
  }
  if (kind === "copy") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16h-9V7h9v14z"/></svg>';
  }
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 7h12l-1 14H7L6 7zm3-4h6l1 2h4v2H4V5h4l1-2z"/></svg>';
}

function createApiActionButton(kind, label, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `api-action-button ${kind}`;
  button.innerHTML = `${apiIconSvg(kind)}<span>${label}</span>`;
  button.addEventListener("click", onClick);
  return button;
}

function renderApiCallList() {
  apiCallListEl.innerHTML = "";
  const count = apiCallsCache.length;
  apiKeySummaryEl.textContent = count === 0 ? "No API keys saved yet." : `${count} saved API key${count === 1 ? "" : "s"}.`;
  if (count === 0) {
    const empty = document.createElement("div");
    empty.className = "api-call-card empty";
    empty.textContent = "No API calls yet. Save one from the builder and it will appear here.";
    apiCallListEl.appendChild(empty);
    return;
  }

  apiCallsCache.forEach((item) => {
    const card = document.createElement("article");
    card.className = "api-call-card";

    const header = document.createElement("div");
    header.className = "api-call-card-header";

    const titleBlock = document.createElement("div");
    titleBlock.className = "api-call-card-title";

    const title = document.createElement("strong");
    title.textContent = item.name;

    const subtitle = document.createElement("span");
    subtitle.textContent = `${maskApiKey(item.api_key)} · ${item.call_id}`;

    titleBlock.append(title, subtitle);

    const actions = document.createElement("div");
    actions.className = "api-call-card-actions";
    actions.append(
      createApiActionButton("copy", "Copy", () => copyTextToClipboard(item.api_key, "API key copied.")),
      createApiActionButton("edit", "Edit", () => editApiCall(item.call_id)),
      createApiActionButton("delete", "Delete", () => {
        pendingApiDeleteId = item.call_id;
        renderDeletePreview(item);
        openModal(apiDeleteModalEl);
      }),
    );

    header.append(titleBlock, actions);

    const meta = document.createElement("div");
    meta.className = "api-call-card-meta";
    [
      ["Saved model", item.model_id],
      ["Embedding", item.embedding_model_id || "Not recorded"],
      ["Internet", item.web_mode],
      ["User context", item.use_user_context ? "enabled" : "disabled"],
      ["Invocation memory", item.instance_mode],
      ["Last used", item.last_invoked_at || "Never"],
      ["Updated", item.updated_at],
    ].forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = "api-call-meta-row";
      const key = document.createElement("span");
      key.textContent = label;
      const detail = document.createElement("strong");
      detail.textContent = value;
      row.append(key, detail);
      meta.appendChild(row);
    });

    card.append(header, meta);
    apiCallListEl.appendChild(card);
  });
}

function resetApiForm() {
  editingApiCallId = null;
  apiFormTitleEl.textContent = "Setup New API Call";
  setApiFormBadge("Draft", false);
  setApiFormStatus("Save a reusable API call definition and Anveshak will generate a dedicated key for it.");
  renderApiKeyBox(null);
  apiNameEl.value = "";
  apiModelEl.value = runtimeStatusSnapshot?.model_id || "Waiting for runtime...";
  apiEmbeddingModelEl.value = runtimeStatusSnapshot?.embedding_model_id || "Waiting for runtime...";
  apiSystemPromptEl.value = "";
  apiInputTemplateEl.value = "User input:\\n{{input}}\\n\\nVariables:\\n{{json}}";
  apiResponseInstructionsEl.value = "";
  apiResponseModeEl.value = "text";
  applyApiWebMode("auto");
  applyUserContextToggle(false);
  applyInstanceMode("independent");
  updateApiUsagePreview();
}

async function editApiCall(callId) {
  showView("api");
  showApiScreen("builder");
  const response = await fetch(`/api/api-calls/${callId}`);
  const payload = await response.json();
  editingApiCallId = payload.call_id;
  apiFormTitleEl.textContent = "Edit API Call";
  setApiFormBadge("Saved", true);
  setApiFormStatus("Editing a saved API call. Saving again also warms the runtime for this workflow.");
  renderApiKeyBox(payload);
  apiNameEl.value = payload.name;
  apiModelEl.value = payload.model_id;
  apiEmbeddingModelEl.value = payload.embedding_model_id || runtimeStatusSnapshot?.embedding_model_id || "Waiting for runtime...";
  apiSystemPromptEl.value = payload.system_prompt;
  apiInputTemplateEl.value = payload.input_template;
  apiResponseInstructionsEl.value = payload.response_instructions;
  apiResponseModeEl.value = payload.response_mode;
  applyApiWebMode(payload.web_mode || "auto");
  applyUserContextToggle(Boolean(payload.use_user_context));
  applyInstanceMode(payload.instance_mode || "independent");
  updateApiUsagePreview(payload);
}

async function saveApiCall() {
  if (!apiNameEl.value.trim()) {
    setApiFormStatus("API call name cannot be empty.", "error");
    return;
  }
  const body = {
    name: apiNameEl.value.trim(),
    system_prompt: apiSystemPromptEl.value,
    input_template: apiInputTemplateEl.value,
    response_instructions: apiResponseInstructionsEl.value,
    response_mode: apiResponseModeEl.value,
    web_mode: currentApiWebMode,
    use_user_context: apiUseUserContext,
    instance_mode: apiInstanceMode,
  };
  const url = editingApiCallId ? `/api/api-calls/${editingApiCallId}` : "/api/api-calls";
  const method = editingApiCallId ? "PUT" : "POST";
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    setApiFormStatus(payload.detail || "The API call could not be saved.", "error");
    return;
  }
  const payload = await response.json();
  editingApiCallId = payload.call_id;
  apiFormTitleEl.textContent = "Edit API Call";
  setApiFormBadge("Saved", true);
  setApiFormStatus("API call saved. Anveshak is warming the runtime for this workflow in the background.", "success");
  renderApiKeyBox(payload);
  apiModelEl.value = payload.model_id;
  apiEmbeddingModelEl.value = payload.embedding_model_id || runtimeStatusSnapshot?.embedding_model_id || "Waiting for runtime...";
  applyApiWebMode(payload.web_mode || "auto");
  applyUserContextToggle(Boolean(payload.use_user_context));
  applyInstanceMode(payload.instance_mode || "independent");
  updateApiUsagePreview(payload);
  await loadApiCalls();
  showApiKeyModal(payload);
}

async function deletePendingApiCall() {
  if (!pendingApiDeleteId) return;
  const response = await fetch(`/api/api-calls/${pendingApiDeleteId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    setApiFormStatus(payload.detail || "The API key could not be deleted.", "error");
    return;
  }
  if (editingApiCallId === pendingApiDeleteId) {
    resetApiForm();
  }
  pendingApiDeleteId = null;
  closeModal(apiDeleteModalEl);
  await loadApiCalls();
  showApiScreen("keys");
  setApiFormStatus("API key deleted.", "success");
}

sendButtonEl.addEventListener("click", sendPrompt);
steerButtonEl.addEventListener("click", sendSteeringNote);
homeButtonEl.addEventListener("click", () => showView("chat"));
newApiCallEl.addEventListener("click", () => {
  showView("api");
  showApiScreen("builder");
});
existingApiCallsEl.addEventListener("click", async () => {
  showView("api");
  showApiScreen("keys");
  await loadApiCalls();
});
apiBuilderTabEl.addEventListener("click", () => showApiScreen("builder"));
apiKeysTabEl.addEventListener("click", async () => {
  showApiScreen("keys");
  await loadApiCalls();
});
apiCreateFromKeysEl.addEventListener("click", () => {
  resetApiForm();
  showApiScreen("builder");
});
saveApiCallEl.addEventListener("click", saveApiCall);
resetApiFormEl.addEventListener("click", resetApiForm);
webModeOptionEls.forEach((option) => {
  option.addEventListener("click", () => applyWebMode(option.dataset.webMode));
});
themeOptionEls.forEach((option) => {
  option.addEventListener("click", () => applyThemePreference(option.dataset.themeMode));
});
apiWebModeOptionEls.forEach((option) => {
  option.addEventListener("click", () => applyApiWebMode(option.dataset.webMode));
});
apiUserContextOptionEls.forEach((option) => {
  option.addEventListener("click", () => applyUserContextToggle(option.dataset.booleanMode === "on"));
});
apiInstanceModeOptionEls.forEach((option) => {
  option.addEventListener("click", () => applyInstanceMode(option.dataset.instanceMode));
});
apiCopyKeyEl.addEventListener("click", () => copyTextToClipboard(apiKeyValueEl.textContent, "API key copied."));
apiOpenKeyDocsEl.addEventListener("click", () => {
  if (!apiKeyValueEl.textContent) return;
  showApiKeyModal({
    call_id: editingApiCallId || "<call_id>",
    api_key: apiKeyValueEl.textContent,
  });
});
apiKeyModalCopyEl.addEventListener("click", () => copyTextToClipboard(apiKeyModalValueEl.textContent, "API key copied."));
apiDeleteConfirmEl.addEventListener("click", deletePendingApiCall);
huggingFaceTokenSubmitEl?.addEventListener("click", submitHuggingFaceToken);
modalCloseEls.forEach((button) => {
  button.addEventListener("click", () => {
    closeModal(document.getElementById(button.dataset.closeModal));
  });
});
[apiKeyModalEl, apiDeleteModalEl, huggingFaceTokenModalEl].forEach((modalEl) => {
  modalEl?.addEventListener("click", (event) => {
    if (event.target === modalEl) {
      closeModal(modalEl);
    }
  });
});

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendPrompt();
  }
});

huggingFaceTokenInputEl?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    submitHuggingFaceToken();
  }
});

fileInputEl.addEventListener("change", () => {
  if (promptLocked) {
    fileInputEl.value = "";
    return;
  }
  pendingFiles.push(...Array.from(fileInputEl.files));
  renderFileChips();
  fileInputEl.value = "";
});

if (systemThemeQuery) {
  const handleSystemThemeChange = () => {
    if (currentThemePreference === "system") {
      applyThemePreference("system", false);
    }
  };

  if (typeof systemThemeQuery.addEventListener === "function") {
    systemThemeQuery.addEventListener("change", handleSystemThemeChange);
  } else if (typeof systemThemeQuery.addListener === "function") {
    systemThemeQuery.addListener(handleSystemThemeChange);
  }
}

applyThemePreference(loadThemePreference(), false);
applyWebMode(currentWebMode);
applyApiWebMode(currentApiWebMode);
applyUserContextToggle(apiUseUserContext);
applyInstanceMode(apiInstanceMode);

["dragenter", "dragover"].forEach((eventName) => {
  dropZoneEl.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (promptLocked) return;
    dropZoneEl.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZoneEl.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZoneEl.classList.remove("dragging");
  });
});

dropZoneEl.addEventListener("drop", (event) => {
  event.preventDefault();
  if (promptLocked) return;
  pendingFiles.push(...Array.from(event.dataTransfer.files));
  renderFileChips();
});

async function boot() {
  resetApiForm();
  await ensureSession();
  await loadApiCalls();
  await refreshRuntimeStatus();
  updateRunControls();
  showApiScreen("builder");
  connectRuntimeStream();
}

boot();
