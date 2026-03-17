let sessionId = null;
let activeRunId = null;
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
const pendingFiles = [];

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
const homeButtonEl = document.getElementById("home-button");
const newApiCallEl = document.getElementById("new-api-call");
const existingApiCallsEl = document.getElementById("existing-api-calls");

const messagesEl = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const fileInputEl = document.getElementById("file-input");
const fileListEl = document.getElementById("file-list");
const sendButtonEl = document.getElementById("send-button");
const steerInputEl = document.getElementById("steer-input");
const steerButtonEl = document.getElementById("steer-button");
const steerBadgeEl = document.getElementById("steer-badge");
const dropZoneEl = document.getElementById("drop-zone");

const apiCallListEl = document.getElementById("api-call-list");
const apiFormTitleEl = document.getElementById("api-form-title");
const apiKeyBoxEl = document.getElementById("api-key-box");
const apiNameEl = document.getElementById("api-name");
const apiModelEl = document.getElementById("api-model");
const apiSystemPromptEl = document.getElementById("api-system-prompt");
const apiInputTemplateEl = document.getElementById("api-input-template");
const apiResponseInstructionsEl = document.getElementById("api-response-instructions");
const apiResponseModeEl = document.getElementById("api-response-mode");
const saveApiCallEl = document.getElementById("save-api-call");
const resetApiFormEl = document.getElementById("reset-api-form");

async function ensureSession() {
  if (sessionId) return sessionId;
  const response = await fetch("/api/sessions", { method: "POST" });
  const payload = await response.json();
  sessionId = payload.session_id;
  return sessionId;
}

function showView(view) {
  chatViewEl.classList.toggle("active", view === "chat");
  apiViewEl.classList.toggle("active", view === "api");
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

function addMessage(role, content) {
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
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { wrapper, body, activity };
}

function updateRunControls() {
  const canSendPrompt = runtimeReady && !promptLocked;
  const canSteer = Boolean(activeRunId) && activeRunPhase === "generation";

  sendButtonEl.disabled = !canSendPrompt;
  promptEl.disabled = !canSendPrompt;
  fileInputEl.disabled = !canSendPrompt;

  steerInputEl.disabled = !canSteer;
  steerButtonEl.disabled = !canSteer;

  if (canSteer) {
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

  addMessage("user", text || "[Attachment only]");
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
  promptLocked = false;
  activeAssistantNode = null;
  activeReasoningNode = null;
  activeActivityFeed = null;
  activeActivityLastText = "";
  updateRunControls();
}

async function sendSteeringNote() {
  const text = steerInputEl.value.trim();
  if (!text || !activeRunId || activeRunPhase !== "generation") return;
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
  steerInputEl.value = "";
}

async function refreshRuntimeStatus() {
  const response = await fetch("/api/runtime/status");
  const payload = await response.json();
  applyRuntimeStatus(payload);
}

function applyRuntimeStatus(payload) {
  runtimeMessageEl.textContent = payload.message;
  runtimeProgressBarEl.style.width = `${Math.round((payload.progress || 0) * 100)}%`;
  runtimePercentEl.textContent = `${Math.round((payload.progress || 0) * 100)}%`;
  runtimeFileEl.textContent = payload.current_file || "Waiting...";
  runtimeModelEl.textContent = payload.model_id;
  runtimeEmbeddingEl.textContent = payload.embedding_model_id;
  renderRuntimeSummary(payload);

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
  apiCallListEl.innerHTML = "";
  if ((payload.items || []).length === 0) {
    const empty = document.createElement("div");
    empty.className = "api-call-item";
    empty.textContent = "No API calls yet.";
    apiCallListEl.appendChild(empty);
    return;
  }
  payload.items.forEach((item) => {
    const node = document.createElement("button");
    node.className = "api-call-item";
    node.innerHTML = `<strong>${item.name}</strong><span>${item.call_id}</span>`;
    node.onclick = () => editApiCall(item.call_id);
    apiCallListEl.appendChild(node);
  });
}

function resetApiForm() {
  editingApiCallId = null;
  apiFormTitleEl.textContent = "Setup New API Call";
  apiKeyBoxEl.classList.add("hidden");
  apiKeyBoxEl.textContent = "";
  apiNameEl.value = "";
  apiModelEl.value = "Uses the current assistant model";
  apiSystemPromptEl.value = "";
  apiInputTemplateEl.value = "User input:\\n{{input}}\\n\\nVariables:\\n{{json}}";
  apiResponseInstructionsEl.value = "";
  apiResponseModeEl.value = "text";
}

async function editApiCall(callId) {
  showView("api");
  const response = await fetch(`/api/api-calls/${callId}`);
  const payload = await response.json();
  editingApiCallId = payload.call_id;
  apiFormTitleEl.textContent = "Edit API Call";
  apiKeyBoxEl.classList.remove("hidden");
  apiKeyBoxEl.textContent = payload.api_key;
  apiNameEl.value = payload.name;
  apiModelEl.value = payload.model_id;
  apiSystemPromptEl.value = payload.system_prompt;
  apiInputTemplateEl.value = payload.input_template;
  apiResponseInstructionsEl.value = payload.response_instructions;
  apiResponseModeEl.value = payload.response_mode;
}

async function saveApiCall() {
  const body = {
    name: apiNameEl.value.trim(),
    system_prompt: apiSystemPromptEl.value,
    input_template: apiInputTemplateEl.value,
    response_instructions: apiResponseInstructionsEl.value,
    response_mode: apiResponseModeEl.value,
  };
  const url = editingApiCallId ? `/api/api-calls/${editingApiCallId}` : "/api/api-calls";
  const method = editingApiCallId ? "PUT" : "POST";
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  editingApiCallId = payload.call_id;
  apiFormTitleEl.textContent = "Edit API Call";
  apiKeyBoxEl.classList.remove("hidden");
  apiKeyBoxEl.textContent = payload.api_key;
  apiModelEl.value = payload.model_id;
  await loadApiCalls();
}

sendButtonEl.addEventListener("click", sendPrompt);
steerButtonEl.addEventListener("click", sendSteeringNote);
homeButtonEl.addEventListener("click", () => showView("chat"));
newApiCallEl.addEventListener("click", () => {
  resetApiForm();
  showView("api");
});
existingApiCallsEl.addEventListener("click", async () => {
  showView("api");
  await loadApiCalls();
});
saveApiCallEl.addEventListener("click", saveApiCall);
resetApiFormEl.addEventListener("click", resetApiForm);

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendPrompt();
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
  updateRunControls();
  connectRuntimeStream();
}

boot();
