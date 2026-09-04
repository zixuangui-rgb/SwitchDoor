"use strict";

const ui = {
  mapping: document.querySelector("#mapping-select"),
  seed: document.querySelector("#seed-input"),
  newSession: document.querySelector("#new-session"),
  frame: document.querySelector("#frame"),
  loading: document.querySelector("#loading-screen"),
  status: document.querySelector("#status-value"),
  previousAction: document.querySelector("#previous-action"),
  actionCount: document.querySelector("#action-count"),
  level: document.querySelector("#level-value"),
  mode: document.querySelector("#mode-value"),
  runnerActions: document.querySelector("#runner-actions"),
  rootSeed: document.querySelector("#root-seed"),
  missionTitle: document.querySelector("#mission-title"),
  missionCopy: document.querySelector("#mission-copy"),
  revealCard: document.querySelector("#reveal-card"),
  ruleSymbol: document.querySelector("#rule-symbol"),
  ruleCopy: document.querySelector("#rule-copy"),
  advance: document.querySelector("#advance-level"),
  hint: document.querySelector("#control-hint"),
  announcement: document.querySelector("#announcement"),
  toast: document.querySelector("#toast"),
  clock: document.querySelector("#clock"),
  levelNodes: [...document.querySelectorAll(".level-node")],
  actionButtons: [...document.querySelectorAll("[data-action]")],
};

const missionText = {
  1: ["发现隐藏映射", "在红色开关上交互，观察哪一扇门真实打开，再抵达目标。"],
  2: ["应用刚才的判断", "本关目标位于红门之后。选择能打开红门的开关。"],
  3: ["在更大布局中迁移", "规则保持不变；只增加导航距离和空间布局。"],
};

const keyActions = {
  ArrowUp: "MOVE_UP",
  w: "MOVE_UP",
  W: "MOVE_UP",
  ArrowDown: "MOVE_DOWN",
  s: "MOVE_DOWN",
  S: "MOVE_DOWN",
  ArrowLeft: "MOVE_LEFT",
  a: "MOVE_LEFT",
  A: "MOVE_LEFT",
  ArrowRight: "MOVE_RIGHT",
  d: "MOVE_RIGHT",
  D: "MOVE_RIGHT",
  " ": "INTERACT",
};

let busy = false;
let levelDone = false;
let episodeDone = false;
let toastTimer = null;

async function request(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || `请求失败 (${response.status})`);
  }
  return body;
}

async function read(path) {
  const response = await fetch(path, { cache: "no-store" });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || `请求失败 (${response.status})`);
  }
  return body;
}

function setBusy(value) {
  busy = value;
  ui.newSession.disabled = value;
  ui.mapping.disabled = value;
  ui.seed.disabled = value;
  ui.actionButtons.forEach((button) => {
    button.disabled = value || levelDone || episodeDone;
  });
  ui.advance.disabled = value;
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  ui.toast.textContent = message;
  ui.toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    ui.toast.hidden = true;
  }, 4200);
}

function announce(message) {
  ui.announcement.textContent = "";
  window.setTimeout(() => {
    ui.announcement.textContent = message;
  }, 20);
}

function mappingReveal(mapping) {
  if (mapping === "same_color") {
    return {
      symbol: "R→R / B→B",
      copy: "同色映射：红开关开启红门，蓝开关开启蓝门。",
    };
  }
  return {
    symbol: "R→B / B→R",
    copy: "异色映射：红开关开启蓝门，蓝开关开启红门。",
  };
}

function render(payload) {
  const observation = payload.observation;
  const runner = payload.runner;
  levelDone = runner.level_done;
  episodeDone = runner.episode_done;

  ui.loading.hidden = true;
  ui.frame.src = observation.image_url;
  ui.frame.alt = `SwitchDoor ${runner.level_id} 当前 RGB 观察`;
  ui.status.textContent = observation.status;
  ui.previousAction.textContent = observation.previous_action || "NONE";
  ui.actionCount.textContent = String(observation.available_actions.length);

  ui.level.textContent = `${runner.level_id} / 3`;
  ui.mode.textContent = runner.session_mode === "blind" ? "BLIND" : "PRACTICE";
  ui.runnerActions.textContent = String(runner.action_count).padStart(2, "0");
  ui.rootSeed.textContent = String(runner.root_seed);
  ui.missionTitle.textContent = missionText[runner.level][0];
  ui.missionCopy.textContent = missionText[runner.level][1];

  ui.levelNodes.forEach((node) => {
    const nodeLevel = Number(node.dataset.level);
    node.classList.toggle("complete", nodeLevel < runner.level || (episodeDone && nodeLevel === 3));
    node.classList.toggle("current", nodeLevel === runner.level && !episodeDone);
  });

  ui.advance.hidden = !runner.can_advance;
  ui.revealCard.hidden = !episodeDone;
  if (episodeDone && runner.revealed_mapping) {
    const reveal = mappingReveal(runner.revealed_mapping);
    ui.ruleSymbol.textContent = reveal.symbol;
    ui.ruleCopy.textContent = reveal.copy;
    ui.hint.textContent = "三关完成。规则已揭示，可启动新回合。";
    announce(`实验完成。${reveal.copy}`);
  } else if (levelDone) {
    ui.hint.textContent = `${runner.level_id} 已完成。确认后进入下一关。`;
    announce(`${runner.level_id} 完成，进入下一关按钮已可用。`);
  } else {
    ui.hint.textContent = "仪器已连接，等待动作输入。";
  }
  setBusy(false);
}

async function startSession() {
  if (busy) return;
  levelDone = false;
  episodeDone = false;
  setBusy(true);
  ui.loading.hidden = false;
  ui.revealCard.hidden = true;
  const payload = { mapping: ui.mapping.value };
  if (ui.seed.value.trim() !== "") {
    const parsed = Number(ui.seed.value);
    if (!Number.isSafeInteger(parsed)) {
      showToast("Seed 必须是安全整数。留空可自动生成。");
      ui.loading.hidden = true;
      setBusy(false);
      return;
    }
    payload.root_seed = parsed;
  }
  try {
    render(await request("/api/session", payload));
    announce("新回合已启动，当前为第一关。");
  } catch (error) {
    ui.loading.hidden = true;
    setBusy(false);
    showToast(error.message);
  }
}

async function sendAction(action, sourceButton = null) {
  if (busy || levelDone || episodeDone) return;
  setBusy(true);
  if (sourceButton) sourceButton.classList.add("pressed");
  try {
    render(await request("/api/action", { action }));
  } catch (error) {
    setBusy(false);
    showToast(error.message);
  } finally {
    if (sourceButton) {
      window.setTimeout(() => sourceButton.classList.remove("pressed"), 100);
    }
  }
}

async function advanceLevel() {
  if (busy || !levelDone || episodeDone) return;
  setBusy(true);
  try {
    const payload = await request("/api/advance", {});
    render(payload);
    announce(`已进入${payload.runner.level_id}。`);
  } catch (error) {
    setBusy(false);
    showToast(error.message);
  }
}

async function resumeOrStart() {
  setBusy(true);
  ui.loading.hidden = false;
  try {
    const payload = await read("/api/state");
    if (!payload.observation) {
      setBusy(false);
      await startSession();
      return;
    }
    render(payload);
    announce("已恢复当前回合。");
  } catch (_error) {
    setBusy(false);
    await startSession();
  }
}

ui.newSession.addEventListener("click", startSession);
ui.advance.addEventListener("click", advanceLevel);
ui.actionButtons.forEach((button) => {
  button.addEventListener("click", () => sendAction(button.dataset.action, button));
});

document.addEventListener("keydown", (event) => {
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement || target instanceof HTMLButtonElement) {
    return;
  }
  const action = keyActions[event.key];
  if (!action) return;
  event.preventDefault();
  const button = ui.actionButtons.find((candidate) => candidate.dataset.action === action);
  sendAction(action, button);
});

function updateClock() {
  const now = new Date();
  ui.clock.dateTime = now.toISOString();
  ui.clock.textContent = now.toLocaleTimeString("zh-CN", { hour12: false });
}

updateClock();
window.setInterval(updateClock, 1000);
resumeOrStart();
