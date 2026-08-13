const cameraCards = new Map(
  [...document.querySelectorAll("[data-camera]")].map((card) => [card.dataset.camera, card]),
);
const frameRequests = new Map();
const frameUrls = new Map();
let latestStatus = null;
let statusOnline = true;

const clamp = (value, minimum = -1, maximum = 1) =>
  Math.min(maximum, Math.max(minimum, Number(value) || 0));

function setDeviceState(element, connected, pending = false) {
  element.classList.toggle("is-online", connected);
  element.classList.toggle("is-error", !connected && !pending);
  const label = element.querySelector("b");
  if (label) label.textContent = connected ? "在线" : pending ? "连接中" : "离线";
}

function updateCamera(camera) {
  const card = cameraCards.get(camera.key) || cameraCards.get(camera.role);
  if (!card) return;
  card.querySelector(".camera-model").textContent = camera.model;
  card.querySelector(".resolution").textContent = `${camera.resolution[0]} x ${camera.resolution[1]}`;
  card.querySelector(".fps").textContent = `${camera.fps.toFixed(1)} FPS`;
  setDeviceState(card.querySelector(".device-state"), camera.connected);
  const placeholder = card.querySelector(".camera-placeholder");
  placeholder.querySelector("strong").textContent = camera.error ? "视频流不可用" : "等待视频流";
  placeholder.querySelector("small").textContent = camera.error || "";
  if (!camera.connected) card.classList.remove("has-frame");
}

function setAxisBar(row, value) {
  const normalized = clamp(value);
  const fill = row.querySelector(".axis-track span");
  const marker = row.querySelector(".axis-track i");
  const magnitude = Math.abs(normalized) * 50;
  fill.style.left = normalized < 0 ? `${50 - magnitude}%` : "50%";
  fill.style.width = `${magnitude}%`;
  marker.style.left = `calc(${50 + normalized * 50}% - 1px)`;
  row.querySelector("output").textContent = `${normalized >= 0 ? "+" : ""}${normalized.toFixed(3)}`;
}

function setVectorField(fieldId, x, y) {
  const dot = document.querySelector(`#${fieldId} .field-dot`);
  dot.style.transform = `translate(${clamp(x) * 45}px, ${clamp(-y) * 45}px)`;
}

function setZMeter(meterId, value) {
  const meter = document.getElementById(meterId);
  const fill = meter.querySelector("span");
  const marker = meter.querySelector("i");
  const normalized = clamp(value);
  const magnitude = Math.abs(normalized) * 50;
  fill.style.top = normalized > 0 ? `${50 - magnitude}%` : "50%";
  fill.style.height = `${magnitude}%`;
  marker.style.top = `calc(${50 - normalized * 50}% - 1px)`;
}

function updateSpaceMouse(mouse) {
  const panel = document.getElementById("spacemouse-panel");
  setDeviceState(document.getElementById("spacemouse-state"), mouse.connected);
  panel.classList.toggle("is-offline", !mouse.connected);
  document.getElementById("spacemouse-device").textContent = mouse.connected
    ? mouse.device
    : mouse.error || "SpaceMouse 未连接";

  const motion = mouse.motion?.length === 6 ? mouse.motion : [0, 0, 0, 0, 0, 0];
  document.querySelectorAll(".axis-row").forEach((row) => {
    setAxisBar(row, motion[Number(row.dataset.axis)]);
  });
  setVectorField("translation-field", motion[0], motion[1]);
  setZMeter("translation-z", motion[2]);
  setVectorField("rotation-field", motion[3], motion[4]);
  setZMeter("rotation-z", motion[5]);

  document.querySelectorAll("[data-button]").forEach((element) => {
    element.classList.toggle("is-pressed", Boolean(mouse.buttons?.[element.dataset.button]));
  });
  const activity = document.getElementById("activity-state");
  activity.classList.toggle("is-active", mouse.active);
  activity.querySelector("b").textContent = mouse.active ? "输入中" : "静止";
  const detail = activity.querySelector("small");
  if (!mouse.connected) detail.textContent = "设备离线";
  else if (mouse.active) detail.textContent = "正在接收六轴输入";
  else if (mouse.last_activity_ms == null) detail.textContent = "等待输入";
  else detail.textContent = `${(mouse.last_activity_ms / 1000).toFixed(1)} 秒前有输入`;
}

function updateSystemState(status) {
  const state = document.getElementById("system-state");
  const mouseOnline = Boolean(status.spacemouse.connected);
  const allOnline = status.camera_online === status.camera_count && mouseOnline;
  const anyOnline = status.camera_online > 0 || mouseOnline;
  state.classList.toggle("is-online", allOnline);
  state.classList.toggle("is-error", !anyOnline);
  document.getElementById("system-state-text").textContent = allOnline
    ? "全部设备在线"
    : `${status.camera_online}/${status.camera_count} 相机 · SpaceMouse ${mouseOnline ? "在线" : "离线"}`;
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    latestStatus = await response.json();
    statusOnline = true;
    latestStatus.cameras.forEach(updateCamera);
    updateSpaceMouse(latestStatus.spacemouse);
    updateSystemState(latestStatus);
  } catch (_error) {
    statusOnline = false;
    const state = document.getElementById("system-state");
    state.classList.remove("is-online");
    state.classList.add("is-error");
    document.getElementById("system-state-text").textContent = "监控服务离线";
  } finally {
    window.setTimeout(pollStatus, document.hidden ? 1000 : 180);
  }
}

async function pollFrame(slot) {
  if (frameRequests.get(slot)) return;
  frameRequests.set(slot, true);
  const card = cameraCards.get(slot);
  try {
    const camera = latestStatus?.cameras?.find((item) => item.key === slot || item.role === slot);
    if (!statusOnline || !camera?.connected) return;
    const cameraKey = encodeURIComponent(camera.key);
    const response = await fetch(`/api/cameras/${cameraKey}/frame.jpg?t=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) return;
    const blob = await response.blob();
    const nextUrl = URL.createObjectURL(blob);
    const image = card.querySelector("img");
    const previousUrl = frameUrls.get(slot);
    image.onload = () => {
      card.classList.add("has-frame");
      if (previousUrl) URL.revokeObjectURL(previousUrl);
    };
    image.src = nextUrl;
    frameUrls.set(slot, nextUrl);
  } catch (_error) {
    card.classList.remove("has-frame");
  } finally {
    frameRequests.set(slot, false);
  }
}

function startFrameLoop(slot) {
  const loop = async () => {
    await pollFrame(slot);
    window.setTimeout(loop, document.hidden ? 700 : 90);
  };
  loop();
}

function updateClock() {
  document.getElementById("local-time").textContent = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date());
}

document.getElementById("refresh-button").addEventListener("click", () => window.location.reload());
updateClock();
window.setInterval(updateClock, 1000);
pollStatus();
cameraCards.forEach((_card, slot) => startFrameLoop(slot));
