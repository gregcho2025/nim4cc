const overviewCards = document.getElementById("overview-cards");
const dashboardUpdated = document.getElementById("dashboard-updated");
const timelineHeader = document.getElementById("timeline-header");
const healthGrid = document.getElementById("health-grid");
const dashboardEmpty = document.getElementById("dashboard-empty");
const refreshDashboardBtn = document.getElementById("refresh-dashboard");

const DISPLAY_TIMEZONE = "Asia/Shanghai";

const dateTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  timeZone: DISPLAY_TIMEZONE,
});

const SUMMARY_WIDTH = 360;
const CELL_SIZE = 42;
const CELL_GAP = 8;

function formatDateTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return dateTimeFormatter.format(date);
}

function rateMeta(rate) {
  if (rate === null || rate === undefined) return { label: "暂无数据", className: "status-cyan", colorClass: "cell-empty" };
  if (rate >= 95) return { label: "优秀", className: "status-cyan", colorClass: "cell-cyan" };
  if (rate >= 80) return { label: "良好", className: "status-green", colorClass: "cell-green" };
  if (rate >= 50) return { label: "告警", className: "status-orange", colorClass: "cell-orange" };
  return { label: "异常", className: "status-red", colorClass: "cell-red" };
}

function createSummaryCard(label, value, detail = "") {
  const card = document.createElement("article");
  card.className = "summary-card";
  card.innerHTML = `<span>${label}</span><strong>${value}</strong><p>${detail}</p>`;
  return card;
}

function applyTimelineGrid(pointsCount) {
  const headerColumns = `minmax(${SUMMARY_WIDTH}px, 1fr) repeat(${pointsCount}, ${CELL_SIZE}px)`;
  timelineHeader.style.gridTemplateColumns = headerColumns;
}

function renderOverview(data) {
  overviewCards.innerHTML = "";
  const averageHealth = data.average_health === null || data.average_health === undefined ? "--" : `${data.average_health.toFixed(2)}%`;
  const totalCalls = data.total_requests ?? 0;
  const healthyModels = (data.models || []).filter((model) => (model.latest_success_rate ?? 0) >= 95).length;
  const displayedBuckets = data.models?.[0]?.points?.length || 0;
  const healthWindowMinutes = data.health_window_minutes || 120;

  overviewCards.appendChild(createSummaryCard("总调用次数", totalCalls, "统计来自网关累计成功与失败请求"));
  overviewCards.appendChild(createSummaryCard("平均健康度", averageHealth, `按监控模型最近 ${healthWindowMinutes} 分钟滚动成功率平均值计算`));
  overviewCards.appendChild(createSummaryCard("高健康模型数", healthyModels, `最近 ${healthWindowMinutes} 分钟滚动成功率达到 95% 以上的模型数量`));
  overviewCards.appendChild(createSummaryCard("统计窗口", `${displayedBuckets * (data.bucket_minutes || 10)} 分钟`, `当前按 ${data.bucket_minutes || 10} 分钟粒度滚动统计`));
  dashboardUpdated.textContent = formatDateTime(data.generated_at);
}

function renderTimelineHeader(models) {
  timelineHeader.innerHTML = "";
  const points = models?.[0]?.points || [];
  applyTimelineGrid(points.length);

  const emptyCell = document.createElement("div");
  emptyCell.className = "timeline-label timeline-left-placeholder";
  timelineHeader.appendChild(emptyCell);

  points.forEach((point) => {
    const cell = document.createElement("div");
    cell.className = "timeline-label";
    cell.textContent = point.label || formatDateTime(point.bucket_start);
    timelineHeader.appendChild(cell);
  });
}

function renderHealthRows(models) {
  healthGrid.innerHTML = "";
  if (!models || models.length === 0) {
    dashboardEmpty.textContent = "当前没有可展示的模型健康数据，请确认 MODEL_LIST 配置并等待请求进入统计。";
    return;
  }
  dashboardEmpty.textContent = "";
  renderTimelineHeader(models);

  models.forEach((model) => {
    const latestRate = model.latest_success_rate === null || model.latest_success_rate === undefined ? "--" : `${model.latest_success_rate.toFixed(2)}%`;
    const latestMeta = rateMeta(model.latest_success_rate);
    const healthWindowMinutes = model.health_window_minutes || 120;

    const row = document.createElement("article");
    row.className = "health-row-card";
    row.style.gridTemplateColumns = `minmax(${SUMMARY_WIDTH}px, 1fr) max-content`;

    const summary = document.createElement("div");
    summary.className = "health-row-summary";
    summary.innerHTML = `
      <div class="health-row-accent ${latestMeta.className}"></div>
      <div class="health-row-copy">
        <h3 title="${model.model_id}">${model.model_id}</h3>
        <div class="health-meta-inline">
          <span class="health-rate-pill ${latestMeta.className}" title="最近 ${healthWindowMinutes} 分钟滚动成功率">${latestRate}</span>
          <span class="health-rate-pill health-call-pill">调用 ${model.total_calls ?? 0} 次</span>
        </div>
      </div>
    `;

    const cells = document.createElement("div");
    cells.className = "health-cells";
    cells.style.gridTemplateColumns = `repeat(${model.points?.length || 0}, ${CELL_SIZE}px)`;
    cells.style.justifySelf = "end";
    (model.points || []).forEach((point) => {
      const meta = rateMeta(point.success_rate);
      const cell = document.createElement("div");
      cell.className = `health-cell ${meta.colorClass}`;
      cell.title = `${point.label} 成功 ${point.success_count}/${point.total_count}`;
      cells.appendChild(cell);
    });

    row.appendChild(summary);
    row.appendChild(cells);
    healthGrid.appendChild(row);
  });
}

async function loadDashboard() {
  const response = await fetch("/api/dashboard", { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error("健康看板数据加载失败");
  }
  const payload = await response.json();
  renderOverview(payload);
  renderHealthRows(payload.models || []);
}

async function refreshDashboard() {
  try {
    await loadDashboard();
  } catch (error) {
    dashboardEmpty.textContent = error.message;
    timelineHeader.innerHTML = "";
    healthGrid.innerHTML = "";
  }
}

refreshDashboardBtn?.addEventListener("click", refreshDashboard);

window.addEventListener("DOMContentLoaded", () => {
  refreshDashboard();
  window.setInterval(refreshDashboard, 60 * 1000);
});
