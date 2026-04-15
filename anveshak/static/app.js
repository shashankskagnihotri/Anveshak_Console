let sessionId = null;
let activeRunId = null;
let activeUserMessageWrapper = null;
let activeAssistantMessage = null;
let activeAssistantNode = null;
let activeReasoningNode = null;
let activeActivityFeed = null;
let activeActivityLastText = "";
let activeRunPhase = null;
let promptLocked = false;
let runtimeReady = false;
let runtimeStream = null;
let runtimeReconnectTimer = null;
let workspaceIndexPollTimer = null;
let editingApiCallId = null;
let currentWebMode = "auto";
let currentMediaMode = "safe";
let currentApiWebMode = "auto";
let currentThemePreference = "system";
let apiUseUserContext = false;
let apiInstanceMode = "independent";
let runtimeStatusSnapshot = null;
let pendingApiDeleteId = null;
let apiCallsCache = [];
let lastHuggingFaceAuthPromptVersion = -1;
let runtimeModelProfile = null;
let runtimeModelSupportsAudio = false;
let micAudioContext = null;
let micMediaStream = null;
let micSourceNode = null;
let micAnalyserNode = null;
let micProcessorNode = null;
let micSilentGainNode = null;
let micAnimationFrame = null;
let micAutoStopTimer = null;
let micRecordedBuffers = [];
let micIsRecording = false;
let micIsProcessing = false;
let micIsTranscribing = false;
let micStatusOverride = "";
let whisperWarmupInFlight = null;
const pendingFiles = [];

const MAX_MIC_RECORDING_MS = 30000;
const MIC_INPUT_BUFFER_SIZE = 4096;

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
const micButtonEl = document.getElementById("mic-button");
const micPanelEl = document.getElementById("mic-panel");
const micVisualizerEl = document.getElementById("mic-visualizer");
const micStatusEl = document.getElementById("mic-status");
const sendButtonEl = document.getElementById("send-button");
const webModeToggleEl = document.getElementById("web-mode-toggle");
const webModeOptionEls = Array.from(document.querySelectorAll(".web-mode-option"));
const mediaModeToggleEl = document.getElementById("media-mode-toggle");
const mediaModeOptionEls = Array.from(document.querySelectorAll(".media-mode-option"));
const mediaModeWarningEl = document.getElementById("media-mode-warning");
const steerInputEl = document.getElementById("steer-input");
const steerButtonEl = document.getElementById("steer-button");
const steerBadgeEl = document.getElementById("steer-badge");
const steerHintEl = document.getElementById("steer-hint");
const workspaceIndexPanelEl = document.getElementById("workspace-index-panel");
const workspaceIndexBadgeEl = document.getElementById("workspace-index-badge");
const workspaceIndexTitleEl = document.getElementById("workspace-index-title");
const workspaceIndexDetailEl = document.getElementById("workspace-index-detail");
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

function normalizeMediaMode(mode) {
  return mode === "unrestricted" ? "unrestricted" : "safe";
}

function normalizeThemePreference(mode) {
  if (mode === "light" || mode === "night") return mode;
  return "system";
}

function resolveRuntimeModelProfile(payload) {
  const items = Array.isArray(payload?.available_models) ? payload.available_models : [];
  const directMatch = items.find((item) => item.model_id === payload?.model_id);
  if (directMatch) return directMatch;

  const lowered = String(payload?.model_id || "").toLowerCase();
  if (lowered.includes("gemma-4-e2b") || lowered.includes("gemma-4-e4b")) {
    return { supports_audio: true };
  }
  if (lowered.includes("gemma-4")) {
    return { supports_audio: false };
  }
  return null;
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

function applyMediaMode(mode) {
  currentMediaMode = normalizeMediaMode(mode);
  if (!mediaModeToggleEl) return;
  mediaModeToggleEl.classList.remove("mode-safe", "mode-unrestricted");
  mediaModeToggleEl.classList.add(`mode-${currentMediaMode}`);
  mediaModeOptionEls.forEach((option) => {
    const selected = option.dataset.mediaMode === currentMediaMode;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-checked", String(selected));
  });
  if (mediaModeWarningEl) {
    mediaModeWarningEl.classList.toggle("hidden", currentMediaMode !== "unrestricted");
  }
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

function fileLooksLikeAudio(file) {
  const mimeType = String(file?.type || "");
  if (mimeType.startsWith("audio/")) return true;
  return /\.(aac|flac|m4a|mp3|ogg|opus|wav)$/i.test(String(file?.name || ""));
}

function describeAttachment(file) {
  const parts = file.name.split(".");
  const extension = parts.length > 1 ? parts.pop().toUpperCase() : "FILE";
  const mimeType = String(file.type || "");
  const isImage = mimeType.startsWith("image/");
  const isAudio = fileLooksLikeAudio(file);
  return {
    name: file.name,
    badge: isAudio ? "AUDIO" : extension || "FILE",
    kind: isImage ? "image" : isAudio ? "audio" : "file",
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

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function configureMarkdownRenderer() {
  if (!window.marked?.setOptions) return;
  window.marked.setOptions({
    gfm: true,
    breaks: true,
    mangle: false,
    headerIds: false,
  });
}

function renderMarkdownHtml(text) {
  const source = escapeHtml(text);
  if (!window.marked?.parse) {
    return source.replace(/\n/g, "<br />");
  }
  return window.marked.parse(source);
}

function isSafeRenderedUrl(url) {
  if (!url) return false;
  try {
    const resolved = new URL(url, window.location.origin);
    return ["http:", "https:", "mailto:"].includes(resolved.protocol);
  } catch (error) {
    return false;
  }
}

function decorateRenderedAssistantBody(body) {
  body.querySelectorAll("img").forEach((image) => {
    const replacement = document.createElement("a");
    replacement.className = "markdown-image-link";
    const imageUrl = image.getAttribute("src") || "";
    replacement.textContent = image.alt ? `Open image: ${image.alt}` : "Open image";
    if (isSafeRenderedUrl(imageUrl)) {
      replacement.href = imageUrl;
      replacement.target = "_blank";
      replacement.rel = "noreferrer noopener";
    } else {
      replacement.removeAttribute("href");
      replacement.setAttribute("aria-disabled", "true");
    }
    image.replaceWith(replacement);
  });
  body.querySelectorAll("a").forEach((link) => {
    const href = link.getAttribute("href") || "";
    if (!isSafeRenderedUrl(href)) {
      link.removeAttribute("href");
      link.setAttribute("aria-disabled", "true");
      return;
    }
    link.target = "_blank";
    link.rel = "noreferrer noopener";
  });
}

function renderLatexIntoBody(body) {
  if (typeof window.renderMathInElement !== "function") return;
  try {
    window.renderMathInElement(body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
      strict: "ignore",
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  } catch (error) {
    // Keep the readable source text when LaTeX rendering fails.
  }
}

function buildAssistantRenderState(body) {
  return {
    body,
    rawText: "",
    markdownEnabled: true,
    finalized: false,
    renderQueued: false,
    toggleEl: null,
    markdownButtonEl: null,
    textButtonEl: null,
  };
}

function updateAssistantRenderToggle(state) {
  if (!state?.toggleEl) return;
  state.toggleEl.classList.toggle("mode-markdown", state.markdownEnabled);
  state.toggleEl.classList.toggle("mode-plain", !state.markdownEnabled);
  if (state.markdownButtonEl) {
    state.markdownButtonEl.classList.toggle("is-selected", state.markdownEnabled);
    state.markdownButtonEl.setAttribute("aria-checked", String(state.markdownEnabled));
  }
  if (state.textButtonEl) {
    state.textButtonEl.classList.toggle("is-selected", !state.markdownEnabled);
    state.textButtonEl.setAttribute("aria-checked", String(!state.markdownEnabled));
  }
}

function renderAssistantBody(state) {
  if (!state?.body) return;
  state.renderQueued = false;
  const rawText = String(state.rawText || "");
  if (!state.markdownEnabled) {
    state.body.classList.remove("markdown-enabled");
    state.body.classList.add("plain-mode");
    state.body.textContent = rawText;
    updateAssistantRenderToggle(state);
    return;
  }

  state.body.classList.remove("plain-mode");
  state.body.classList.add("markdown-enabled");
  state.body.innerHTML = renderMarkdownHtml(rawText);
  decorateRenderedAssistantBody(state.body);
  if (state.finalized) {
    renderLatexIntoBody(state.body);
  }
  updateAssistantRenderToggle(state);
}

function scheduleAssistantRender(state) {
  if (!state || state.renderQueued) return;
  state.renderQueued = true;
  window.requestAnimationFrame(() => {
    renderAssistantBody(state);
  });
}

function setAssistantMarkdownMode(state, enabled) {
  if (!state) return;
  state.markdownEnabled = Boolean(enabled);
  renderAssistantBody(state);
}

function appendAssistantText(message, text) {
  if (!message?.renderState) {
    if (message?.body) message.body.textContent += text;
    return;
  }
  message.renderState.rawText += String(text || "");
  scheduleAssistantRender(message.renderState);
}

function finalizeAssistantMessage(message) {
  if (!message?.renderState) return;
  message.renderState.finalized = true;
  renderAssistantBody(message.renderState);
}

function buildAssistantRenderToggle(state) {
  const toggle = document.createElement("div");
  toggle.className = "message-render-toggle mode-markdown";
  toggle.setAttribute("role", "radiogroup");
  toggle.setAttribute("aria-label", "Assistant response rendering");

  const thumb = document.createElement("span");
  thumb.className = "message-render-thumb";
  thumb.setAttribute("aria-hidden", "true");

  const markdownButton = document.createElement("button");
  markdownButton.type = "button";
  markdownButton.className = "message-render-option is-selected";
  markdownButton.textContent = "MD";
  markdownButton.setAttribute("role", "radio");
  markdownButton.setAttribute("aria-checked", "true");
  markdownButton.title = "Render Markdown and LaTeX";
  markdownButton.addEventListener("click", () => setAssistantMarkdownMode(state, true));

  const textButton = document.createElement("button");
  textButton.type = "button";
  textButton.className = "message-render-option";
  textButton.textContent = "TXT";
  textButton.setAttribute("role", "radio");
  textButton.setAttribute("aria-checked", "false");
  textButton.title = "Show raw text and raw Markdown";
  textButton.addEventListener("click", () => setAssistantMarkdownMode(state, false));

  toggle.append(thumb, markdownButton, textButton);
  state.toggleEl = toggle;
  state.markdownButtonEl = markdownButton;
  state.textButtonEl = textButton;
  return toggle;
}

function addMessage(role, content, options = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const header = document.createElement("div");
  header.className = "message-header";

  const label = document.createElement("div");
  label.className = "label";
  label.textContent = role === "user" ? "You" : "Anveshak";
  header.append(label);

  const activity = document.createElement("div");
  activity.className = "activity-feed hidden";

  const body = document.createElement("div");
  body.className = "content";
  const renderState = role === "assistant" ? buildAssistantRenderState(body) : null;
  if (renderState) {
    renderState.rawText = String(content || "");
    header.append(buildAssistantRenderToggle(renderState));
  } else {
    body.textContent = content;
  }

  wrapper.append(header);
  if (role === "assistant") {
    wrapper.append(activity);
  }
  wrapper.append(body);
  appendAttachmentPreviews(wrapper, options.attachments || []);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  if (renderState) {
    renderAssistantBody(renderState);
  }
  return { wrapper, body, activity, renderState };
}

function addMicrophoneTranscriptionStatusMessage(attachmentName) {
  const wrapper = document.createElement("div");
  wrapper.className = "message transcription-status";

  const label = document.createElement("div");
  label.className = "label";
  label.textContent = "Anveshak";

  const card = document.createElement("div");
  card.className = "transcription-status-card is-live";

  const visual = document.createElement("div");
  visual.className = "transcription-status-visual";
  for (let index = 0; index < 5; index += 1) {
    const bar = document.createElement("span");
    bar.className = "transcription-bar";
    bar.style.setProperty("--bar-index", String(index));
    visual.appendChild(bar);
  }

  const copy = document.createElement("div");
  copy.className = "transcription-status-copy";

  const title = document.createElement("div");
  title.className = "transcription-status-title";
  title.textContent = "Transcribing";

  const detail = document.createElement("div");
  detail.className = "transcription-status-detail";
  detail.textContent = attachmentName
    ? `Whisper is turning ${attachmentName} into editable chat text.`
    : "Whisper is turning your microphone recording into editable chat text.";

  const preview = document.createElement("div");
  preview.className = "transcription-status-preview hidden";

  copy.append(title, detail, preview);
  card.append(visual, copy);
  wrapper.append(label, card);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return { wrapper, card, title, detail, preview };
}

function updateMicrophoneTranscriptionStatus(cardState, { tone = "live", title = "", detail = "", preview = "" } = {}) {
  if (!cardState?.card) return;
  cardState.card.classList.remove("is-live", "is-success", "is-error");
  cardState.card.classList.add(
    tone === "success" ? "is-success" : tone === "error" ? "is-error" : "is-live",
  );
  if (title) cardState.title.textContent = title;
  if (detail) cardState.detail.textContent = detail;
  if (preview) {
    cardState.preview.textContent = preview;
    cardState.preview.classList.remove("hidden");
  } else {
    cardState.preview.textContent = "";
    cardState.preview.classList.add("hidden");
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function mergeTranscriptIntoPrompt(text) {
  const transcript = String(text || "").trim();
  if (!transcript) return;
  const existing = promptEl.value.trim();
  promptEl.value = existing ? `${existing}\n${transcript}` : transcript;
  promptEl.focus();
  promptEl.selectionStart = promptEl.value.length;
  promptEl.selectionEnd = promptEl.value.length;
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

function appendAudioTranscription(messageWrapper, payload) {
  if (!messageWrapper || !payload?.text) return;

  let container = messageWrapper.querySelector(".audio-transcriptions");
  if (!container) {
    container = document.createElement("div");
    container.className = "audio-transcriptions";
    messageWrapper.appendChild(container);
  }

  const card = document.createElement("div");
  card.className = "audio-transcription";

  const tag = document.createElement("div");
  tag.className = "audio-transcription-tag";
  const backendLabel = payload.backend ? ` • ${payload.backend}` : "";
  tag.textContent = payload.attachment_name ? `Transcribed audio${backendLabel} • ${payload.attachment_name}` : `Transcribed audio${backendLabel}`;

  const body = document.createElement("div");
  body.className = "audio-transcription-text";
  body.textContent = payload.text;

  card.append(tag, body);
  container.appendChild(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function summarizeUserMessage(text, attachments) {
  if (text) return text;
  if (attachments.some((attachment) => attachment.kind === "audio")) return "[Audio message]";
  return "[Attachment only]";
}

function updateRunControls() {
  const canSendPrompt = runtimeReady && !promptLocked && !micIsRecording && !micIsProcessing && !micIsTranscribing;
  const canSteer = Boolean(activeRunId) && activeRunPhase === "generation";

  sendButtonEl.disabled = !canSendPrompt;
  promptEl.disabled = false;
  fileInputEl.disabled = !canSendPrompt;
  syncMicrophoneUi();

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

function setMicrophoneStatus(text) {
  if (micStatusEl) micStatusEl.textContent = text;
}

function fillRoundedRect(context, x, y, width, height, radius) {
  const cornerRadius = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + cornerRadius, y);
  context.lineTo(x + width - cornerRadius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + cornerRadius);
  context.lineTo(x + width, y + height - cornerRadius);
  context.quadraticCurveTo(x + width, y + height, x + width - cornerRadius, y + height);
  context.lineTo(x + cornerRadius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - cornerRadius);
  context.lineTo(x, y + cornerRadius);
  context.quadraticCurveTo(x, y, x + cornerRadius, y);
  context.closePath();
  context.fill();
}

function resetMicrophoneVisualizer() {
  if (!micVisualizerEl) return;
  const context = micVisualizerEl.getContext("2d");
  if (!context) return;

  const width = micVisualizerEl.width;
  const height = micVisualizerEl.height;
  context.clearRect(0, 0, width, height);

  context.fillStyle = "rgba(11, 109, 103, 0.12)";
  fillRoundedRect(context, 0, 0, width, height, 14);

  const barCount = 20;
  const gap = 4;
  const horizontalPadding = 10;
  const availableWidth = width - (horizontalPadding * 2);
  const barWidth = (availableWidth - (gap * (barCount - 1))) / barCount;
  const barHeight = 10;
  const y = (height - barHeight) / 2;

  context.fillStyle = "rgba(11, 109, 103, 0.32)";
  for (let index = 0; index < barCount; index += 1) {
    const x = horizontalPadding + (index * (barWidth + gap));
    fillRoundedRect(context, x, y, Math.max(2, barWidth), barHeight, 3);
  }
}

function stopMicrophoneVisualizer() {
  if (micAnimationFrame) {
    window.cancelAnimationFrame(micAnimationFrame);
    micAnimationFrame = null;
  }
  resetMicrophoneVisualizer();
}

function startMicrophoneVisualizer() {
  if (!micAnalyserNode || !micVisualizerEl) return;
  const context = micVisualizerEl.getContext("2d");
  if (!context) return;

  const width = micVisualizerEl.width;
  const height = micVisualizerEl.height;
  const sampleBuffer = new Uint8Array(micAnalyserNode.fftSize);
  const barCount = 20;
  const gap = 4;
  const horizontalPadding = 10;
  const availableWidth = width - (horizontalPadding * 2);
  const barWidth = (availableWidth - (gap * (barCount - 1))) / barCount;

  const drawFrame = () => {
    if (!micAnalyserNode || !micIsRecording) {
      stopMicrophoneVisualizer();
      return;
    }

    micAnalyserNode.getByteTimeDomainData(sampleBuffer);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "rgba(177, 63, 63, 0.1)";
    fillRoundedRect(context, 0, 0, width, height, 14);

    const samplesPerBar = Math.max(1, Math.floor(sampleBuffer.length / barCount));
    for (let barIndex = 0; barIndex < barCount; barIndex += 1) {
      let magnitude = 0;
      const start = barIndex * samplesPerBar;
      const end = Math.min(sampleBuffer.length, start + samplesPerBar);
      for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
        magnitude += Math.abs((sampleBuffer[sampleIndex] - 128) / 128);
      }
      magnitude /= Math.max(1, end - start);

      const barHeight = Math.max(6, Math.min(height - 8, 8 + (magnitude * (height - 10) * 2.8)));
      const x = horizontalPadding + (barIndex * (barWidth + gap));
      const y = height - barHeight - 4;
      context.fillStyle = magnitude > 0.32 ? "rgba(177, 63, 63, 0.92)" : "rgba(177, 63, 63, 0.52)";
      fillRoundedRect(context, x, y, Math.max(2, barWidth), barHeight, 4);
    }

    micAnimationFrame = window.requestAnimationFrame(drawFrame);
  };

  drawFrame();
}

function syncMicrophoneUi() {
  if (!micButtonEl || !micPanelEl) return;

  micButtonEl.classList.remove("hidden");
  micPanelEl.classList.remove("hidden");
  micButtonEl.classList.toggle("is-recording", micIsRecording);
  micPanelEl.classList.toggle("is-recording", micIsRecording);
  micPanelEl.classList.toggle("is-processing", micIsProcessing);
  micPanelEl.classList.toggle("is-transcribing", micIsTranscribing);
  micButtonEl.setAttribute("aria-pressed", String(micIsRecording));

  if (micIsRecording) {
    setMicrophoneStatus("Recording audio... tap the mic again to stop.");
  } else if (micIsProcessing) {
    setMicrophoneStatus("Finalizing audio clip...");
  } else if (micIsTranscribing) {
    setMicrophoneStatus("Whisper is transcribing your recording into editable text...");
  } else if (micStatusOverride) {
    setMicrophoneStatus(micStatusOverride);
  } else if (runtimeModelSupportsAudio) {
    setMicrophoneStatus("Mic ready. Whisper handles live voice, and Gemma can transcribe attached audio.");
  } else {
    setMicrophoneStatus("Mic ready. Whisper will transcribe your voice.");
  }
  if (!micIsRecording) {
    resetMicrophoneVisualizer();
  }

  micButtonEl.title = micIsRecording ? "Stop recording" : "Record audio";
  micButtonEl.disabled = micIsProcessing || micIsTranscribing || promptLocked || (!runtimeReady && !micIsRecording);
}

async function cleanupMicrophoneResources() {
  if (micAutoStopTimer) {
    window.clearTimeout(micAutoStopTimer);
    micAutoStopTimer = null;
  }

  stopMicrophoneVisualizer();

  if (micProcessorNode) {
    micProcessorNode.onaudioprocess = null;
    micProcessorNode.disconnect();
    micProcessorNode = null;
  }
  if (micAnalyserNode) {
    micAnalyserNode.disconnect();
    micAnalyserNode = null;
  }
  if (micSourceNode) {
    micSourceNode.disconnect();
    micSourceNode = null;
  }
  if (micSilentGainNode) {
    micSilentGainNode.disconnect();
    micSilentGainNode = null;
  }

  if (micMediaStream) {
    micMediaStream.getTracks().forEach((track) => track.stop());
    micMediaStream = null;
  }

  if (micAudioContext) {
    try {
      await micAudioContext.close();
    } catch (error) {
      // Closing the audio context is best effort.
    }
    micAudioContext = null;
  }
}

function flattenMicrophoneBuffers(buffers) {
  const totalLength = buffers.reduce((sum, buffer) => sum + buffer.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  buffers.forEach((buffer) => {
    merged.set(buffer, offset);
    offset += buffer.length;
  });
  return merged;
}

function writeWaveString(view, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

function encodeWaveFile(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + (samples.length * 2));
  const view = new DataView(buffer);

  writeWaveString(view, 0, "RIFF");
  view.setUint32(4, 36 + (samples.length * 2), true);
  writeWaveString(view, 8, "WAVE");
  writeWaveString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeWaveString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return buffer;
}

function buildMicrophoneFilename() {
  return `microphone-recording-${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
}

function buildRecordedAudioFile(buffers, sampleRate) {
  if (!buffers.length) return null;
  const merged = flattenMicrophoneBuffers(buffers);
  if (!merged.length || merged.length < Math.max(1200, Math.floor(sampleRate * 0.2))) {
    return null;
  }
  const wavBytes = encodeWaveFile(merged, sampleRate || 16000);
  return new File([wavBytes], buildMicrophoneFilename(), { type: "audio/wav" });
}

async function startMicrophoneRecording() {
  if (micIsRecording || micIsProcessing || micIsTranscribing) return;
  micStatusOverride = "";
  requestWhisperWarmup();
  if (!navigator.mediaDevices?.getUserMedia) {
    addWarningMessage("This browser cannot access the microphone.");
    return;
  }

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    addWarningMessage("This browser does not expose the Web Audio APIs needed for live microphone capture.");
    return;
  }

  try {
    micMediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  } catch (error) {
    addWarningMessage("Microphone permission was denied or unavailable.");
    return;
  }

  micRecordedBuffers = [];
  micAudioContext = new AudioContextClass();
  if (micAudioContext.state === "suspended") {
    await micAudioContext.resume().catch(() => {});
  }

  micSourceNode = micAudioContext.createMediaStreamSource(micMediaStream);
  micAnalyserNode = micAudioContext.createAnalyser();
  micAnalyserNode.fftSize = 2048;
  micAnalyserNode.smoothingTimeConstant = 0.82;
  micProcessorNode = micAudioContext.createScriptProcessor(MIC_INPUT_BUFFER_SIZE, 1, 1);
  micSilentGainNode = micAudioContext.createGain();
  micSilentGainNode.gain.value = 0;

  micSourceNode.connect(micAnalyserNode);
  micSourceNode.connect(micProcessorNode);
  micProcessorNode.connect(micSilentGainNode);
  micSilentGainNode.connect(micAudioContext.destination);
  micProcessorNode.onaudioprocess = (event) => {
    const channelData = event.inputBuffer.getChannelData(0);
    if (!channelData?.length) return;
    micRecordedBuffers.push(new Float32Array(channelData));
  };

  micIsRecording = true;
  micAutoStopTimer = window.setTimeout(() => {
    stopMicrophoneRecording({ autoStopped: true }).catch(() => {
      addWarningMessage("The microphone recording could not be finalized cleanly.");
    });
  }, MAX_MIC_RECORDING_MS);

  syncMicrophoneUi();
  startMicrophoneVisualizer();
  updateRunControls();
}

async function stopMicrophoneRecording({ autoStopped = false } = {}) {
  if (!micIsRecording) return;

  micIsRecording = false;
  micIsProcessing = true;
  syncMicrophoneUi();
  updateRunControls();

  const sampleRate = micAudioContext?.sampleRate || 16000;
  const recordedBuffers = micRecordedBuffers.slice();
  await cleanupMicrophoneResources();
  micRecordedBuffers = [];

  const recordedFile = buildRecordedAudioFile(recordedBuffers, sampleRate);
  micIsProcessing = false;
  syncMicrophoneUi();
  updateRunControls();

  if (!recordedFile) {
    setMicrophoneStatus("No usable audio was captured. Try again.");
    return;
  }

  await transcribeMicrophoneRecording(recordedFile, { autoStopped });
}

async function toggleMicrophoneRecording() {
  if (micIsTranscribing) return;
  if (micIsRecording) {
    await stopMicrophoneRecording();
    return;
  }
  await startMicrophoneRecording();
}

function requestWhisperWarmup() {
  if (whisperWarmupInFlight) return whisperWarmupInFlight;
  whisperWarmupInFlight = fetch("/api/runtime/whisper-warmup", { method: "POST" })
    .catch(() => null)
    .finally(() => {
      whisperWarmupInFlight = null;
    });
  return whisperWarmupInFlight;
}

async function transcribeMicrophoneRecording(recordedFile, { autoStopped = false } = {}) {
  await ensureSession();
  micIsTranscribing = true;
  syncMicrophoneUi();
  updateRunControls();

  const statusCard = addMicrophoneTranscriptionStatusMessage(recordedFile.name);
  updateMicrophoneTranscriptionStatus(statusCard, {
    tone: "live",
    title: "Transcribing",
    detail: autoStopped
      ? "Whisper is transcribing your 30-second microphone clip so you can edit it before sending."
      : "Whisper is transcribing your microphone clip so you can edit it before sending.",
  });

  const form = new FormData();
  form.append("file", recordedFile);

  try {
    const response = await fetch(`/api/sessions/${sessionId}/microphone-transcription`, {
      method: "POST",
      body: form,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Whisper could not transcribe the microphone recording.");
    }

    mergeTranscriptIntoPrompt(payload.text || "");
    micStatusOverride = "Transcript ready. Review it in the chat box before sending.";
    updateMicrophoneTranscriptionStatus(statusCard, {
      tone: "success",
      title: "Transcription ready",
      detail: "Inserted into the chat box. Review it, edit it, and send when you're happy.",
      preview: payload.text || "",
    });
  } catch (error) {
    pendingFiles.push(recordedFile);
    renderFileChips();
    micStatusOverride = "Transcription failed. The audio clip was attached instead.";
    updateMicrophoneTranscriptionStatus(statusCard, {
      tone: "error",
      title: "Transcription failed",
      detail: `${error.message || "Whisper could not transcribe the microphone recording."} The audio clip was attached instead so you can still send it manually.`,
    });
  } finally {
    micIsTranscribing = false;
    syncMicrophoneUi();
    updateRunControls();
  }
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
  if (!runtimeReady || promptLocked || micIsRecording || micIsProcessing || micIsTranscribing) return;
  const text = promptEl.value.trim();
  if (!text && pendingFiles.length === 0) return;
  await ensureSession();
  micStatusOverride = "";
  promptLocked = true;
  activeRunPhase = "submit";
  updateRunControls();

  const messageAttachments = pendingFiles.map((file) => describeAttachment(file));
  const userMessage = addMessage("user", summarizeUserMessage(text, messageAttachments), { attachments: messageAttachments });
  activeUserMessageWrapper = userMessage.wrapper;
  const assistant = addMessage("assistant", "");
  activeAssistantMessage = assistant;
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
  form.append("media_mode", currentMediaMode);
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
  source.addEventListener("transcription", (event) => {
    const payload = JSON.parse(event.data).payload;
    appendAudioTranscription(activeUserMessageWrapper, payload);
    pushActivity(`Transcribed ${payload.attachment_name || "audio"} into chat text`, "transcription");
  });
  source.addEventListener("token", (event) => {
    const payload = JSON.parse(event.data).payload;
    appendAssistantText(activeAssistantMessage, payload.text);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
  source.addEventListener("done", (event) => {
    const payload = JSON.parse(event.data).payload;
    finalizeAssistantMessage(activeAssistantMessage);
    appendMediaResults(messageWrapper, payload.media_results || [], payload.media_warning || "");
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

function appendMediaResults(messageWrapper, mediaResults, warningText = "") {
  if (!messageWrapper) return;

  const existing = messageWrapper.querySelector(".web-media-section");
  if (existing) existing.remove();
  if ((!Array.isArray(mediaResults) || mediaResults.length === 0) && !warningText) return;

  const section = document.createElement("section");
  section.className = "web-media-section";

  if (warningText) {
    const warning = document.createElement("div");
    warning.className = "web-media-warning";
    warning.textContent = warningText;
    section.appendChild(warning);
  }

  if (Array.isArray(mediaResults) && mediaResults.length > 0) {
    const header = document.createElement("div");
    header.className = "web-media-header";

    const title = document.createElement("div");
    title.className = "web-media-title";
    title.textContent = "Web media";

    const caption = document.createElement("div");
    caption.className = "web-media-caption";
    caption.textContent = "Inline previews from the live web search";

    header.append(title, caption);
    section.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "web-media-grid";
    mediaResults.forEach((item) => {
      grid.appendChild(createMediaCard(item));
    });
    section.appendChild(grid);
  }

  messageWrapper.appendChild(section);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function createMediaCard(item) {
  const card = document.createElement("article");
  const kind = item?.kind === "video" ? "video" : "image";
  card.className = `web-media-card ${kind}`;

  const preview = createMediaPreview(item, kind);
  card.appendChild(preview);

  const meta = document.createElement("div");
  meta.className = "web-media-meta";

  const tagRow = document.createElement("div");
  tagRow.className = "web-media-tag-row";

  const kindTag = document.createElement("span");
  kindTag.className = "web-media-tag";
  kindTag.textContent = kind === "video" ? "Video" : "Image";

  const sourceTag = document.createElement("span");
  sourceTag.className = "web-media-source";
  sourceTag.textContent = item?.source_label || "Web";

  tagRow.append(kindTag, sourceTag);

  const title = document.createElement("div");
  title.className = "web-media-card-title";
  title.textContent = item?.title || (kind === "video" ? "Web video" : "Web image");

  const snippet = document.createElement("div");
  snippet.className = "web-media-snippet";
  snippet.textContent = item?.snippet || "";

  const actions = document.createElement("div");
  actions.className = "web-media-actions";
  actions.appendChild(
    buildMediaLink(item?.content_url || item?.page_url, kind === "video" ? "Open video" : "Open image"),
  );
  if (item?.page_url && item.page_url !== item.content_url) {
    actions.appendChild(buildMediaLink(item.page_url, "Source page"));
  }

  meta.append(tagRow, title);
  if (item?.snippet) meta.appendChild(snippet);
  meta.appendChild(actions);
  card.appendChild(meta);
  return card;
}

function createMediaPreview(item, kind) {
  const wrap = document.createElement("div");
  wrap.className = "web-media-preview";

  if (kind === "video" && canEmbedVideo(item)) {
    const frame = document.createElement("iframe");
    frame.className = "web-media-embed";
    frame.src = item.embed_url;
    frame.title = item?.title || "Embedded web video";
    frame.loading = "lazy";
    frame.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
    frame.referrerPolicy = "strict-origin-when-cross-origin";
    frame.allowFullscreen = true;
    wrap.appendChild(frame);
    return wrap;
  }

  if (kind === "video" && isDirectVideoUrl(item?.content_url)) {
    const video = document.createElement("video");
    video.className = "web-media-video";
    video.controls = true;
    video.preload = "metadata";
    video.src = item.content_url;
    if (item?.preview_url) video.poster = item.preview_url;
    wrap.appendChild(video);
    return wrap;
  }

  if (item?.preview_url) {
    const image = document.createElement("img");
    image.className = "web-media-image";
    image.src = item.preview_url;
    image.alt = item?.title || (kind === "video" ? "Video preview" : "Image preview");
    image.loading = "lazy";
    wrap.appendChild(image);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "web-media-placeholder";
    placeholder.textContent = kind === "video" ? "Video preview unavailable" : "Image preview unavailable";
    wrap.appendChild(placeholder);
  }

  if (kind === "video") {
    const badge = document.createElement("span");
    badge.className = "web-media-play-badge";
    badge.textContent = "Play";
    wrap.appendChild(badge);
  }
  return wrap;
}

function buildMediaLink(url, label) {
  const link = document.createElement("a");
  link.className = "web-media-link";
  link.href = url || "#";
  link.target = "_blank";
  link.rel = "noreferrer noopener";
  link.textContent = label;
  if (!url) link.setAttribute("aria-disabled", "true");
  return link;
}

function canEmbedVideo(item) {
  const embedUrl = String(item?.embed_url || "");
  return embedUrl.startsWith("https://www.youtube-nocookie.com/embed/")
    || embedUrl.startsWith("https://www.youtube.com/embed/");
}

function isDirectVideoUrl(url) {
  return /\.(mp4|m4v|mov|webm|ogg)(\?|$)/i.test(String(url || ""));
}

function cleanupRun(source) {
  if (source) source.close();
  activeRunId = null;
  activeRunPhase = null;
  activeUserMessageWrapper = null;
  activeAssistantMessage = null;
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

  runtimeModelProfile = resolveRuntimeModelProfile(payload);
  runtimeModelSupportsAudio = Boolean(runtimeModelProfile?.supports_audio);
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

function renderWorkspaceIndexStatus(payload) {
  if (!workspaceIndexPanelEl) return;
  const enabled = Boolean(payload?.enabled);
  const active = Boolean(payload?.active);
  if (!enabled || !active) {
    workspaceIndexPanelEl.classList.add("hidden");
    return;
  }

  workspaceIndexPanelEl.classList.remove("hidden");
  if (workspaceIndexBadgeEl) {
    workspaceIndexBadgeEl.textContent = "Refreshing";
    workspaceIndexBadgeEl.className = "badge active";
  }
  if (workspaceIndexTitleEl) {
    workspaceIndexTitleEl.textContent = payload.message || "Refreshing local-file index";
  }
  if (workspaceIndexDetailEl) {
    workspaceIndexDetailEl.textContent = payload.detail
      || "Anveshak is updating workspace retrieval in the background. You can keep chatting while this runs.";
  }
}

async function refreshWorkspaceIndexStatus() {
  if (!workspaceIndexPanelEl) return;
  try {
    const response = await fetch("/api/workspace-index/status");
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      workspaceIndexPanelEl.classList.add("hidden");
      return;
    }
    renderWorkspaceIndexStatus(payload);
  } catch (error) {
    workspaceIndexPanelEl.classList.add("hidden");
  }
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
micButtonEl?.addEventListener("click", () => {
  toggleMicrophoneRecording().catch(() => {
    addWarningMessage("The microphone recording could not be completed.");
    micIsRecording = false;
    micIsProcessing = false;
    micIsTranscribing = false;
    cleanupMicrophoneResources().catch(() => {});
    syncMicrophoneUi();
    updateRunControls();
  });
});
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
mediaModeOptionEls.forEach((option) => {
  option.addEventListener("click", () => applyMediaMode(option.dataset.mediaMode));
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
  if (promptLocked || micIsRecording || micIsProcessing || micIsTranscribing) {
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
configureMarkdownRenderer();
applyWebMode(currentWebMode);
applyMediaMode(currentMediaMode);
applyApiWebMode(currentApiWebMode);
applyUserContextToggle(apiUseUserContext);
applyInstanceMode(apiInstanceMode);

["dragenter", "dragover"].forEach((eventName) => {
  dropZoneEl.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (promptLocked || micIsRecording || micIsProcessing || micIsTranscribing) return;
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
  if (promptLocked || micIsRecording || micIsProcessing || micIsTranscribing) return;
  pendingFiles.push(...Array.from(event.dataTransfer.files));
  renderFileChips();
});

async function boot() {
  resetApiForm();
  await ensureSession();
  await loadApiCalls();
  await refreshRuntimeStatus();
  await refreshWorkspaceIndexStatus();
  resetMicrophoneVisualizer();
  syncMicrophoneUi();
  updateRunControls();
  showApiScreen("builder");
  connectRuntimeStream();
  if (workspaceIndexPollTimer) {
    window.clearInterval(workspaceIndexPollTimer);
  }
  workspaceIndexPollTimer = window.setInterval(() => {
    refreshWorkspaceIndexStatus().catch(() => {});
  }, 1500);
}

boot();
