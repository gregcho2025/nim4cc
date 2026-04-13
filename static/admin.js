const PANEL_ATTR = "data-panel";
const sidebarButtons = document.querySelectorAll(".sidebar-btn");
const panels = document.querySelectorAll(`.glass-panel[${PANEL_ATTR}]`);
const loginOverlay = document.getElementById("login-overlay");
const loginBtn = document.getElementById("login-btn");
const loginStatus = document.getElementById("login-status");
const overviewMetrics = document.getElementById("overview-metrics");
const recentChecks = document.getElementById("recent-checks");
const modelTable = document.getElementById("model-table");
const keyTable = document.getElementById("key-table");
const healthGrid = document.getElementById("health-grid");
const modelCount = document.getElementById("model-count");
const modelHealthy = document.getElementById("model-healthy");
const settingsStatus = document.getElementById("settings-status");
const testAllModelsBtn = document.getElementById("test-all-models");
const testAllKeysBtn = document.getElementById("test-all-keys");
const runHealthcheckBtn = document.getElementById("run-healthcheck");

const state = {
  token: sessionStorage.getItem("nim_token"),
  panel: "overview",
};

const STATUS_LABELS = {
  healthy: "正常",
  degraded: "波动",
  down: "异常",
  unknown: "未巡检",
};

const dateTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});

function formatDateTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return dateTimeFormatter.format(date);
}

function showPanel(name) {
  panels.forEach((panel) => panel.classList.toggle("hidden", panel.getAttribute(PANEL_ATTR) !== name));
  sidebarButtons.forEach((button) => button.classList.toggle("active", button.dataset.panel === name));
  state.panel = name;
}

sidebarButtons.forEach((button) => button.addEventListener("click", () => showPanel(button.dataset.panel)));

async function apiRequest(endpoint, opts = {}) {
  const headers = { "Content-Type": "application/json", Accept: "application/json" };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(`/admin/api/${endpoint}`, {
    ...opts,
    headers: { ...headers, ...(opts.headers || {}) },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.message || payload.detail || payload.error?.message || "请求失败");
  }
  return response.json();
}

function metricCard(label, value, detail = "") {
  const div = document.createElement("div");
  div.className = "metric-card";
  div.innerHTML = `<h3>${label}</h3><strong>${value}</strong><p>${detail}</p>`;
  return div;
}

function pill(status) {
  const normalized = status || "unknown";
  return `<span class="pill ${normalized}">${STATUS_LABELS[normalized] || normalized}</span>`;
}

async function renderOverview() {
  const payload = await apiRequest("overview");
  const totals = payload.totals || {};
  overviewMetrics.innerHTML = "";
  overviewMetrics.appendChild(metricCard("启用模型", totals.enabled_models ?? "--", `总数 ${totals.total_models ?? "--"}`));
  overviewMetrics.appendChild(metricCard("启用 Key", totals.enabled_keys ?? "--", `总数 ${totals.total_keys ?? "--"}`));
  overviewMetrics.appendChild(metricCard("代理请求", totals.total_requests ?? "--", `成功 ${totals.total_success ?? 0}`));
  overviewMetrics.appendChild(metricCard("失败次数", totals.total_failures ?? "--", "累计转发失败或上游返回错误"));

  recentChecks.innerHTML = "";
  (payload.recent_checks || []).forEach((check) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${formatDateTime(check.time)}</td>
      <td>${check.model}</td>
      <td>${pill(check.status)}</td>
      <td>${check.latency ? `${check.latency} ms` : "--"}</td>
    `;
    recentChecks.appendChild(row);
  });
}

async function renderModels() {
  const payload = await apiRequest("models");
  const items = payload.items || [];
  modelCount.textContent = items.length;
  modelHealthy.textContent = items.filter((item) => item.status === "healthy").length;
  modelTable.innerHTML = "";

  items.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>
        <strong>${item.display_name || item.model_id}</strong><br />
        <span class="status-text mono">${item.model_id}</span>
      </td>
      <td>${pill(item.status)}</td>
      <td>${item.request_count}</td>
      <td>${item.healthcheck_success_count}/${item.healthcheck_count}</td>
      <td>
        <div class="inline-actions">
          <button class="secondary-btn" data-action="test-model" data-id="${item.model_id}">测试</button>
          <button class="secondary-btn danger-btn" data-action="remove-model" data-id="${item.model_id}">删除</button>
        </div>
      </td>
    `;
    modelTable.appendChild(row);
  });
}

async function renderKeys() {
  const payload = await apiRequest("keys");
  const items = payload.items || [];
  keyTable.innerHTML = "";

  items.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.label}</td>
      <td class="mono">${item.masked_key}</td>
      <td>${item.request_count}</td>
      <td>${formatDateTime(item.last_tested)}</td>
      <td>${pill(item.status)}</td>
      <td>
        <div class="inline-actions">
          <button class="secondary-btn" data-action="test-key" data-id="${item.name}">测试</button>
          <button class="secondary-btn danger-btn" data-action="remove-key" data-id="${item.name}">删除</button>
        </div>
      </td>
    `;
    keyTable.appendChild(row);
  });
}

async function renderHealth() {
  const payload = await apiRequest("healthchecks");
  const items = payload.items || [];
  healthGrid.innerHTML = "";

  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-card";
    empty.textContent = "暂无巡检记录，执行一次巡检后这里会显示最新结果。";
    healthGrid.appendChild(empty);
    return;
  }

  items.slice(0, 12).forEach((item) => {
    const card = document.createElement("article");
    card.className = "health-record";
    card.innerHTML = `
      <div class="toolbar-row">
        <div>
          <h4>${item.model}</h4>
          <span class="status-text mono">${item.model_id}</span>
        </div>
        ${pill(item.status)}
      </div>
      <p class="status-text">${item.detail || "暂无详情"}</p>
      <div class="record-meta">
        <span>Key: ${item.api_key || "未记录"}</span>
        <span>时延: ${item.latency ? `${item.latency} ms` : "--"}</span>
        <span>时间: ${formatDateTime(item.checked_at)}</span>
      </div>
    `;
    healthGrid.appendChild(card);
  });
}

async function renderSettings() {
  const payload = await apiRequest("settings");
  document.getElementById("healthcheck-enabled").checked = Boolean(payload.healthcheck_enabled);
  document.getElementById("healthcheck-interval").value = payload.healthcheck_interval_minutes || 60;
  document.getElementById("public-history-hours").value = payload.public_history_hours || 48;
  document.getElementById("healthcheck-prompt").value = payload.healthcheck_prompt || "请只回复 OK。";
}

async function loadAll() {
  await Promise.all([renderOverview(), renderModels(), renderKeys(), renderHealth(), renderSettings()]);
}

async function runAllModelChecks() {
  const payload = await apiRequest("healthchecks/run", { method: "POST", body: JSON.stringify({}) });
  const items = payload.items || [];
  const success = items.filter((item) => item.status === "healthy").length;
  alert(`已完成全部模型巡检，共 ${items.length} 个模型，其中 ${success} 个正常。`);
  showPanel("health");
  await loadAll();
}

async function runAllKeyChecks() {
  const payload = await apiRequest("keys/test-all", { method: "POST", body: JSON.stringify({}) });
  const items = payload.items || [];
  const success = items.filter((item) => item.status === "healthy").length;
  alert(`已完成全部 Key 测试，共 ${items.length} 个 Key，其中 ${success} 个正常。`);
  showPanel("keys");
  await loadAll();
}

async function testModel(modelId) {
  const payload = await apiRequest(`models/${encodeURIComponent(modelId)}/test`, { method: "POST", body: JSON.stringify({}) });
  alert(`${payload.display_name || payload.model} 当前状态：${STATUS_LABELS[payload.status] || payload.status}`);
  await loadAll();
}

async function removeModel(modelId) {
  await apiRequest("models/remove", { method: "POST", body: JSON.stringify({ value: modelId }) });
  await loadAll();
}

async function testKey(keyName) {
  const payload = await apiRequest("keys/test", { method: "POST", body: JSON.stringify({ value: keyName }) });
  alert(`Key ${payload.api_key} 当前状态：${STATUS_LABELS[payload.status] || payload.status}`);
  await loadAll();
}

async function removeKey(keyName) {
  await apiRequest("keys/remove", { method: "POST", body: JSON.stringify({ value: keyName }) });
  await loadAll();
}

modelTable.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  if (button.dataset.action === "test-model") testModel(button.dataset.id);
  if (button.dataset.action === "remove-model") removeModel(button.dataset.id);
});

keyTable.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  if (button.dataset.action === "test-key") testKey(button.dataset.id);
  if (button.dataset.action === "remove-key") removeKey(button.dataset.id);
});

document.getElementById("model-add")?.addEventListener("click", async () => {
  const modelId = document.getElementById("model-id").value.trim();
  const displayName = document.getElementById("model-display-name").value.trim();
  const description = document.getElementById("model-description").value.trim();
  if (!modelId) {
    alert("请先填写模型 ID。");
    return;
  }
  await apiRequest("models", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId, display_name: displayName || modelId, description }),
  });
  document.getElementById("model-id").value = "";
  document.getElementById("model-display-name").value = "";
  document.getElementById("model-description").value = "";
  await renderModels();
});

document.getElementById("key-add")?.addEventListener("click", async () => {
  const name = document.getElementById("key-label").value.trim();
  const apiKey = document.getElementById("key-value").value.trim();
  if (!name || !apiKey) {
    alert("请填写 Key 名称和内容。");
    return;
  }
  await apiRequest("keys", { method: "POST", body: JSON.stringify({ name, api_key: apiKey }) });
  document.getElementById("key-label").value = "";
  document.getElementById("key-value").value = "";
  await renderKeys();
});

testAllModelsBtn?.addEventListener("click", runAllModelChecks);
testAllKeysBtn?.addEventListener("click", runAllKeyChecks);
runHealthcheckBtn?.addEventListener("click", runAllModelChecks);

document.getElementById("settings-save")?.addEventListener("click", async () => {
  try {
    const payload = {
      healthcheck_enabled: document.getElementById("healthcheck-enabled").checked,
      healthcheck_interval_minutes: Number(document.getElementById("healthcheck-interval").value || 60),
      public_history_hours: Number(document.getElementById("public-history-hours").value || 48),
      healthcheck_prompt: document.getElementById("healthcheck-prompt").value.trim(),
    };
    await apiRequest("settings", { method: "PUT", body: JSON.stringify(payload) });
    settingsStatus.textContent = "设置已保存。";
    await loadAll();
  } catch (error) {
    settingsStatus.textContent = error.message;
  }
});

document.getElementById("refresh-now")?.addEventListener("click", loadAll);

loginBtn.addEventListener("click", async () => {
  const password = document.getElementById("admin-password").value.trim();
  if (!password) {
    loginStatus.textContent = "请输入后台密码。";
    return;
  }
  try {
    loginStatus.textContent = "正在验证身份...";
    const response = await fetch("/admin/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ password }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || payload.message || "登录失败");
    state.token = payload.access_token || payload.token;
    sessionStorage.setItem("nim_token", state.token);
    loginOverlay.classList.add("hidden");
    await loadAll();
  } catch (error) {
    loginStatus.textContent = error.message;
  }
});

window.addEventListener("DOMContentLoaded", async () => {
  showPanel(state.panel);
  if (!state.token) return;
  loginOverlay.classList.add("hidden");
  try {
    await loadAll();
    setInterval(loadAll, 90 * 1000);
  } catch (_error) {
    sessionStorage.removeItem("nim_token");
    loginOverlay.classList.remove("hidden");
  }
});
