const catalogSummary = document.getElementById("catalog-summary");
const providerFilterBar = document.getElementById("provider-filter-bar");
const providerGrid = document.getElementById("provider-grid");
const catalogUpdated = document.getElementById("catalog-updated");
const catalogEmpty = document.getElementById("catalog-empty");

let catalogState = {
  providers: [],
  activeProvider: "all",
};

const DISPLAY_TIMEZONE = "Asia/Shanghai";

const dateTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  timeZone: DISPLAY_TIMEZONE,
});

function formatDateTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return dateTimeFormatter.format(date);
}

function createSummaryCard(label, value, detail = "") {
  const card = document.createElement("article");
  card.className = "summary-card";
  card.innerHTML = `<span>${label}</span><strong>${value}</strong><p>${detail}</p>`;
  return card;
}

function renderSummary(data) {
  catalogSummary.innerHTML = "";
  catalogSummary.appendChild(createSummaryCard("官方模型总数", data.total_models ?? 0, "来自 NVIDIA 官方 /v1/models"));
  catalogSummary.appendChild(createSummaryCard("提供商数量", data.providers?.length ?? 0, "按模型 ID 中的提供商前缀自动归类"));
}

function renderFilterBar() {
  providerFilterBar.innerHTML = "";
  const options = [{ provider: "all", count: catalogState.providers.reduce((sum, item) => sum + (item.count || 0), 0), label: "全部" }].concat(
    catalogState.providers.map((group) => ({ provider: group.provider, count: group.count, label: group.provider }))
  );

  options.forEach((option) => {
    const button = document.createElement("button");
    button.className = `provider-filter-btn ${catalogState.activeProvider === option.provider ? "active" : ""}`;
    button.type = "button";
    button.textContent = `${option.label} (${option.count})`;
    button.addEventListener("click", () => {
      catalogState.activeProvider = option.provider;
      renderFilterBar();
      renderProviders();
    });
    providerFilterBar.appendChild(button);
  });
}

function renderProviders() {
  providerGrid.innerHTML = "";
  const filtered = catalogState.activeProvider === "all"
    ? catalogState.providers
    : catalogState.providers.filter((group) => group.provider === catalogState.activeProvider);

  if (filtered.length === 0) {
    catalogEmpty.textContent = "当前筛选条件下没有可展示的模型。";
    return;
  }
  catalogEmpty.textContent = "";

  filtered.forEach((group) => {
    const card = document.createElement("article");
    card.className = "provider-card provider-card-fixed";
    card.innerHTML = `
      <div class="provider-card-head">
        <div>
          <h3>${group.provider}</h3>
          <p>${group.count} 个模型</p>
        </div>
      </div>
    `;

    const body = document.createElement("div");
    body.className = "provider-card-body";

    const list = document.createElement("div");
    list.className = "provider-model-list";
    (group.models || []).forEach((model) => {
      const chip = document.createElement("div");
      chip.className = "provider-model-chip";
      chip.title = model.id;
      chip.textContent = model.id;
      list.appendChild(chip);
    });

    body.appendChild(list);
    card.appendChild(body);
    providerGrid.appendChild(card);
  });
}

async function loadCatalog() {
  const response = await fetch("/api/catalog", { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error("模型列表加载失败");
  }
  const payload = await response.json();
  catalogUpdated.textContent = formatDateTime(payload.synced_at || payload.generated_at);
  catalogState.providers = payload.providers || [];
  renderSummary(payload);
  renderFilterBar();
  renderProviders();
}

window.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadCatalog();
  } catch (error) {
    catalogEmpty.textContent = error.message;
    providerGrid.innerHTML = "";
    providerFilterBar.innerHTML = "";
  }
});
