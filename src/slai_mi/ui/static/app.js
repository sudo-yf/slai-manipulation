const CAMERA_ORDER = ["secondary", "primary", "wrist"];
const CAMERA_LABELS = {
  secondary: "CAM 01 - LEFT",
  primary: "CAM 00 - CENTER MAIN",
  wrist: "CAM 02 - RIGHT",
};
const cameraFigures = new Map();
const cameraPreviews = new Map();

function sortedCameras(cameras) {
  return [...cameras].sort((left, right) => {
    const leftIndex = CAMERA_ORDER.indexOf(left.key);
    const rightIndex = CAMERA_ORDER.indexOf(right.key);
    return (leftIndex < 0 ? 99 : leftIndex) - (rightIndex < 0 ? 99 : rightIndex);
  });
}

function ensureCameras(cameras) {
  const grid = document.getElementById("camera-grid");
  const template = document.getElementById("camera-template");
  for (const camera of sortedCameras(cameras)) {
    let figure = cameraFigures.get(camera.key);
    if (!figure) {
      figure = template.content.firstElementChild.cloneNode(true);
      figure.dataset.camera = camera.key;
      const image = figure.querySelector("img");
      image.dataset.camera = camera.key;
      image.alt = `${CAMERA_LABELS[camera.key] || camera.label || camera.key} live RGB feed`;
      grid.appendChild(figure);
      cameraFigures.set(camera.key, figure);
      cameraPreviews.set(camera.key, createCameraPreview(figure));
    }
    figure.querySelector(".camera-label").textContent =
      CAMERA_LABELS[camera.key] || camera.label || camera.role || camera.key;
    const resolution = Array.isArray(camera.resolution) ? camera.resolution.join("x") : "--";
    const fps = Number(camera.fps || 0);
    figure.querySelector(".camera-format").textContent = `${resolution} / ${fps.toFixed(0)}fps`;
  }
  for (const [key, figure] of cameraFigures) {
    if (!cameras.some((camera) => camera.key === key)) {
      cameraPreviews.get(key)?.stop();
      cameraPreviews.delete(key);
      figure.remove();
      cameraFigures.delete(key);
    }
  }
  const orderedFigures = sortedCameras(cameras)
    .map((camera) => cameraFigures.get(camera.key))
    .filter(Boolean);
  const currentFigures = [...grid.children];
  if (orderedFigures.some((figure, index) => currentFigures[index] !== figure)) {
    grid.replaceChildren(...orderedFigures);
  }
}

function whepUrl(role) {
  const encoded = encodeURIComponent(role);
  if (window.location.hostname === "record.leai.me") {
    return `/${encoded}/whep`;
  }
  const host = window.location.hostname.includes(":")
    ? `[${window.location.hostname}]`
    : window.location.hostname;
  return `${window.location.protocol}//${host}:8889/${encoded}/whep`;
}

function isLanPage() {
  const host = window.location.hostname;
  if (host === "localhost" || host === "::1" || host.startsWith("127.")) return true;
  if (host.startsWith("10.") || host.startsWith("192.168.")) return true;
  const match = host.match(/^172\.(\d+)\./);
  return Boolean(match && Number(match[1]) >= 16 && Number(match[1]) <= 31);
}

function createCameraPreview(figure) {
  const image = figure.querySelector("img");
  const video = figure.querySelector("video");
  const role = image.dataset.camera;
  let peer = null;
  let request = null;
  let retryTimer = null;
  let snapshotTimer = null;
  let retryDelay = 250;
  let stopped = false;
  let generation = 0;

  const setFrame = (visible, webrtc) => {
    figure.dataset.frame = String(visible);
    figure.dataset.webrtc = String(webrtc);
  };
  const loadSnapshot = () => {
    if (stopped || document.hidden) return;
    image.src = `/api/cameras/${encodeURIComponent(role)}/frame.jpg?t=${Date.now()}`;
  };
  const scheduleRetry = () => {
    if (stopped || document.hidden || retryTimer != null) return;
    const delay = retryDelay;
    retryDelay = Math.min(retryDelay * 2, 2000);
    retryTimer = window.setTimeout(() => {
      retryTimer = null;
      connectWebRTC();
    }, delay);
  };
  const scheduleSnapshot = () => {
    if (stopped || document.hidden || snapshotTimer != null) return;
    snapshotTimer = window.setTimeout(() => {
      snapshotTimer = null;
      loadSnapshot();
    }, 250);
  };
  const closePeer = () => {
    generation += 1;
    request?.abort();
    request = null;
    if (peer) {
      peer.onconnectionstatechange = null;
      peer.close();
      peer = null;
    }
    video.srcObject = null;
  };
  const connectWebRTC = async () => {
    if (stopped || document.hidden) return;
    closePeer();
    const currentGeneration = generation;
    if (!("RTCPeerConnection" in window)) {
      loadSnapshot();
      return;
    }
    try {
      const connection = new RTCPeerConnection({
        iceServers: isLanPage() ? [] : [{urls: "stun:stun.l.google.com:19302"}],
      });
      peer = connection;
      const showDecodedFrame = () => {
        const showVideo = () => {
          retryDelay = 250;
          setFrame(true, true);
        };
        if ("requestVideoFrameCallback" in video) {
          video.requestVideoFrameCallback(showVideo);
        } else {
          showVideo();
        }
      };
      const startPlayback = () => {
        video.play().then(showDecodedFrame).catch(() => {});
      };
      video.onloadedmetadata = startPlayback;
      video.onplaying = showDecodedFrame;
      connection.addTransceiver("video", {direction: "recvonly"});
      connection.ontrack = (event) => {
        if (event.streams[0]) video.srcObject = event.streams[0];
      };
      connection.onconnectionstatechange = () => {
        if (peer === connection && ["failed", "disconnected", "closed"].includes(connection.connectionState)) {
          setFrame(false, false);
          loadSnapshot();
          scheduleRetry();
        }
      };
      const offer = await connection.createOffer();
      await connection.setLocalDescription(offer);
      await new Promise((resolve) => {
        if (connection.iceGatheringState === "complete") return resolve();
        const done = () => {
          if (connection.iceGatheringState === "complete") {
            connection.removeEventListener("icegatheringstatechange", done);
            resolve();
          }
        };
        connection.addEventListener("icegatheringstatechange", done);
        window.setTimeout(resolve, isLanPage() ? 350 : 1500);
      });
      if (currentGeneration !== generation) return;
      const controller = new AbortController();
      request = controller;
      const response = await fetch(whepUrl(role), {
        method: "POST",
        headers: {"Content-Type": "application/sdp"},
        body: connection.localDescription.sdp,
        signal: controller.signal,
      });
      if (request === controller) request = null;
      if (currentGeneration !== generation) return;
      if (!response.ok) throw new Error(`WHEP HTTP ${response.status}`);
      await connection.setRemoteDescription({type: "answer", sdp: await response.text()});
      if (video.readyState >= 1) startPlayback();
    } catch (_) {
      if (stopped || document.hidden || currentGeneration !== generation) return;
      closePeer();
      setFrame(false, false);
      loadSnapshot();
      scheduleRetry();
    }
  };
  image.addEventListener("load", () => {
    if (snapshotTimer != null) {
      window.clearTimeout(snapshotTimer);
      snapshotTimer = null;
    }
    if (figure.dataset.webrtc !== "true") setFrame(true, false);
  });
  image.addEventListener("error", () => {
    if (figure.dataset.webrtc !== "true") setFrame(false, false);
    scheduleSnapshot();
  });
  video.addEventListener("error", () => {
    setFrame(false, false);
    loadSnapshot();
    scheduleRetry();
  });
  loadSnapshot();
  connectWebRTC();

  return {
    start() {
      if (!stopped) return;
      stopped = false;
      retryDelay = 250;
      loadSnapshot();
      connectWebRTC();
    },
    stop() {
      stopped = true;
      if (retryTimer != null) window.clearTimeout(retryTimer);
      if (snapshotTimer != null) window.clearTimeout(snapshotTimer);
      retryTimer = null;
      snapshotTimer = null;
      closePeer();
      setFrame(false, false);
    },
  };
}

function renderCameraStatus(cameras) {
  ensureCameras(cameras);
  for (const camera of cameras) {
    const figure = cameraFigures.get(camera.key);
    if (!figure) continue;
    const online = Boolean(camera.connected && camera.valid && camera.age_ms != null);
    const slow = online && Number(camera.age_ms) >= 100;
    figure.dataset.online = String(online);
    if (!online) figure.dataset.frame = "false";
    figure.dataset.slow = String(slow);
    figure.querySelector(".camera-latency").textContent = online
      ? `Lat: ${Number(camera.age_ms).toFixed(0)}ms`
      : "Lat: OFFLINE";
    const resolution = Array.isArray(camera.resolution) ? camera.resolution.join("x") : "--";
    figure.querySelector(".camera-format").textContent =
      `${resolution} / ${Number(camera.fps || 0).toFixed(0)}fps`;
  }
  const onlineCount = cameras.filter((camera) => camera.connected && camera.valid).length;
  document.getElementById("system-label").textContent =
    `Sys: ${onlineCount === cameras.length && cameras.length ? "Nominal" : `${onlineCount}/${cameras.length} Cameras`}`;
}

function eventTime(value) {
  const text = String(value || "--:--:--");
  if (text.includes("T")) return text.split("T")[1].slice(0, 12);
  return text.slice(-12);
}

function renderEvents(events) {
  const list = document.querySelector(".event-list");
  const shouldFollow = list.scrollHeight - list.scrollTop - list.clientHeight < 24;
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "event empty";
    empty.textContent = "Waiting for collection events.";
    list.replaceChildren(empty);
    return;
  }
  list.replaceChildren(...events.map((event) => {
    const row = document.createElement("div");
    row.className = "event";
    row.dataset.level = event.level || "info";
    const timestamp = document.createElement("span");
    timestamp.className = "event-time";
    timestamp.textContent = eventTime(event.time);
    const level = document.createElement("span");
    level.className = "event-level";
    level.textContent = event.code || event.level || "info";
    const message = document.createElement("span");
    message.className = "event-message";
    message.textContent = event.message || "";
    row.append(timestamp, level, message);
    return row;
  }));
  if (shouldFollow) list.scrollTop = list.scrollHeight;
}

let lastEventSignature = "";

function renderRecordingStatus(status) {
  const indicator = document.querySelector(".record-indicator");
  indicator.dataset.state = status.phase || "starting";
  const episode = status.episode || {};
  const detail = status.recording
    ? `${episode.valid_frames || 0} 帧 · ${Number(episode.elapsed_s || 0).toFixed(1)} 秒`
    : (status.can_record ? "等待 MENU" : "");
  document.getElementById("record-label").textContent = detail
    ? `${status.phase_label || "准备中"} · ${detail}`
    : (status.phase_label || "准备中");
  const temperature = status.temperature || {};
  const temperatureBox = document.querySelector(".temperature-status");
  temperatureBox.dataset.level = String(temperature.level || "unknown");
  temperatureBox.querySelector("strong").textContent = temperature.max_c == null
    ? "--"
    : `${Number(temperature.max_c).toFixed(1)}°C`;
  const events = Array.isArray(status.events) ? [...status.events].reverse() : [];
  renderEvents(events);
}

function renderDevices(devices) {
  const names = {ur5: "UR5", wuji: "WujiHand", wrist: "Wrist"};
  const labels = {active: "ON", inactive: "OFF", starting: "STARTING", error: "ERROR"};
  document.getElementById("device-summary").textContent = ["ur5", "wuji", "wrist"]
    .map((name) => `${names[name]} ${labels[devices?.[name]?.state] || "ERROR"}`)
    .join(" · ");
}

async function loadSpaceMouseMap() {
  const host = document.getElementById("spacemouse-container");
  try {
    const response = await fetch("/spacemouse-input-map.svg?v=20260823-4");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    host.innerHTML = await response.text();
    const svg = host.querySelector("svg");
    if (svg) {
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.setAttribute("focusable", "false");
      requestAnimationFrame(positionSpaceMousePuck);
    }
  } catch (_) {
    host.textContent = "Input map unavailable";
  }
}

function positionSpaceMousePuck() {
  const wrap = document.querySelector(".spacemouse-wrap");
  const host = document.getElementById("spacemouse-container");
  const puck = document.getElementById("spacemouse-puck");
  const svg = host?.querySelector("svg");
  if (!wrap || !puck || !svg) return;
  const wrapRect = wrap.getBoundingClientRect();
  const svgRect = svg.getBoundingClientRect();
  // center-ring is cx=252, cy=268 in a 508x502 viewBox.
  puck.style.left = `${svgRect.left - wrapRect.left + svgRect.width * (252 / 508)}px`;
  puck.style.top = `${svgRect.top - wrapRect.top + svgRect.height * (268 / 502)}px`;
}

function setSpaceMouseButtons(buttons) {
  const aliases = {rear: "r", front: "f"};
  for (const button of document.querySelectorAll("#spacemouse-container [data-button]")) {
    const name = button.dataset.button;
    button.dataset.pressed = String(Boolean(buttons?.[name] || buttons?.[aliases[name]]));
  }
}

function setMotion(motion) {
  const values = Array.from({length: 6}, (_, index) =>
    Math.max(-1, Math.min(1, Number(motion?.[index] || 0))));
  const puck = document.getElementById("spacemouse-puck");
  // Keep input values unchanged, but make small real movements visible
  // inside the center ring.
  const travel = 44;
  puck.style.transform = `translate(calc(-50% + ${values[1] * travel}px), calc(-50% + ${values[0] * travel}px)) ` +
    `scale(${1 + values[2] * 0.1}) rotateX(${-values[3] * 18}deg) ` +
    `rotateY(${values[4] * 18}deg) rotateZ(${values[5] * 22}deg)`;
  puck.classList.toggle("active", values.some((value) => Math.abs(value) > 0.01));
  for (const [index, value] of values.entries()) {
    document.querySelector(`[data-axis="${index}"] output`).value = value.toFixed(2);
  }
}

function renderMouse(state) {
  const stateBox = document.querySelector(".spacemouse-state");
  stateBox.dataset.connected = String(Boolean(state.connected && state.valid));
  stateBox.title = state.error || "Read-only status from the SLAI collection process";
  document.getElementById("spacemouse-connection-state").textContent = state.connected
    ? (state.active ? "SpaceMouse Pro · ACTIVE" : "SpaceMouse Pro")
    : "SpaceMouse Pro · WAITING";
  document.getElementById("spacemouse-age").textContent = state.age_ms == null
    ? "--"
    : `${Number(state.age_ms).toFixed(0)}ms`;
  setMotion(state.motion);
  setSpaceMouseButtons(state.buttons);
}

function renderStatus(status) {
  renderCameraStatus(Array.isArray(status.cameras) ? status.cameras : []);
  const eventSignature = JSON.stringify([
    status.phase,
    status.recording,
    status.episode,
    status.temperature,
    status.events,
  ]);
  if (eventSignature !== lastEventSignature) {
    lastEventSignature = eventSignature;
    renderRecordingStatus(status);
  }
  renderDevices(status.devices || {});
}

window.addEventListener("resize", positionSpaceMousePuck, {passive: true});
document.addEventListener("visibilitychange", () => {
  for (const preview of cameraPreviews.values()) {
    if (document.hidden) preview.stop();
    else preview.start();
  }
});
window.addEventListener("pagehide", () => {
  for (const preview of cameraPreviews.values()) preview.stop();
});

let pollingStarted = false;

function startStatusPolling() {
  if (pollingStarted) return;
  pollingStarted = true;
  const poll = async () => {
    try {
      const response = await fetch("/api/status", {cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      renderStatus(await response.json());
    } catch (_) {
      document.getElementById("system-label").textContent = "Sys: Offline";
    } finally {
      window.setTimeout(poll, 250);
    }
  };
  poll();
}

function connectStatusStream() {
  if (!("EventSource" in window)) {
    startStatusPolling();
    return;
  }
  const stream = new EventSource("/api/events");
  let received = false;
  const fallbackTimer = window.setTimeout(() => {
    if (!received) {
      stream.close();
      startStatusPolling();
    }
  }, 4000);
  stream.onmessage = (event) => {
    received = true;
    renderStatus(JSON.parse(event.data));
  };
  stream.onerror = () => {
    if (received) document.getElementById("system-label").textContent = "Sys: Reconnecting";
  };
}

function connectSpaceMouseStream() {
  if (!("EventSource" in window)) return;
  const stream = new EventSource("/api/spacemouse/events");
  stream.onmessage = (event) => {
    renderMouse(JSON.parse(event.data));
  };
  stream.onerror = () => {};
}

loadSpaceMouseMap();
connectStatusStream();
connectSpaceMouseStream();
