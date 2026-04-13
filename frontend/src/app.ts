/**
 * KDA: сбор событий клавиатуры для регистрации и верификации по почерку.
 */

type DevicePreset = "desktop" | "mobile";
type AppMode = "enroll" | "verify";

const CONFIG = {
  userIdPlaceholder: "local-user",
  targetPhrase: "The quick brown fox jumps over the lazy dog",
  requiredAttempts: 30,
  defaultDevicePreset: "desktop" as DevicePreset,
  /**
   * URL бэкенда. Пример: "http://localhost:8000"
   * Пустая строка — не отправлять, только превью в <details>.
   */
  apiBaseUrl: "http://localhost:8000",
  enrollmentPath: "/security/enroll",
  verifyPath: "/security/verify",
};

// ─── Интерфейсы payload для бекенда ───────────────────────────────────────────

interface BackendAttempt {
  attemptId: string;
  events: CapturedEvent[];
}

interface EnrollPayload {
  login: string;
  device_type: DevicePreset;
  phrase: string;
  attempts: BackendAttempt[];
}

interface VerifyPayload {
  login: string;
  device_type: DevicePreset;
  phrase: string;
  attempt: BackendAttempt;
}

// ─── Внутренние интерфейсы ─────────────────────────────────────────────────────

interface AttemptRecord {
  attemptId: string;
  startedAt: number;
  endedAt: number;
  targetText: string;
  finalText: string;
  typing_speed: number;
  events: CapturedEvent[];
}

type CapturedEvent =
  | { type: "focus"; t: number }
  | { type: "blur"; t: number }
  | { type: "paste"; t: number }
  | {
      type: "keydown" | "keyup";
      key: string;
      code: string;
      location: number;
      t: number;
      repeat: boolean;
      value: string;
      caretStart: number;
      caretEnd: number;
    }
  | {
      type: "beforeinput";
      inputType: string;
      data: string | null;
      t: number;
      value: string;
      caretStart: number;
      caretEnd: number;
    }
  | {
      type: "input";
      inputType: string;
      data: string | null;
      value: string;
      caretStart: number;
      caretEnd: number;
      t: number;
    }
  | { type: "compositionstart"; t: number; data: string }
  | { type: "compositionupdate"; t: number; data: string }
  | { type: "compositionend"; t: number; data: string };

// ─── Helpers ───────────────────────────────────────────────────────────────────

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id}`);
  return el;
}

function caretSnapshot(el: HTMLTextAreaElement): {
  value: string;
  caretStart: number;
  caretEnd: number;
} {
  return {
    value: el.value,
    caretStart: el.selectionStart ?? 0,
    caretEnd: el.selectionEnd ?? 0,
  };
}

function nowMs(): number {
  return performance.now();
}

// ─── AttemptCapture ────────────────────────────────────────────────────────────

class AttemptCapture {
  private enrollmentT0: number;
  private attemptPerf0: number | null = null;
  private events: CapturedEvent[] = [];
  private ended = false;

  constructor(enrollmentT0: number) {
    this.enrollmentT0 = enrollmentT0;
  }

  startAttemptWall(perfNow: number): void {
    if (this.attemptPerf0 !== null) return;
    this.attemptPerf0 = perfNow;
  }

  localT(perfNow: number): number {
    if (this.attemptPerf0 === null) return 0;
    return Math.round((perfNow - this.attemptPerf0) * 1000) / 1000;
  }

  startedAtSession(): number {
    if (this.attemptPerf0 === null) return 0;
    return Math.round((this.attemptPerf0 - this.enrollmentT0) * 1000) / 1000;
  }

  endedAtSession(perfNow: number): number {
    return Math.round((perfNow - this.enrollmentT0) * 1000) / 1000;
  }

  push(e: CapturedEvent, perfNow: number): void {
    if (this.ended) return;
    if (this.attemptPerf0 === null && e.type !== "focus") {
      this.startAttemptWall(perfNow);
    }
    this.events.push(e);
  }

  finish(): CapturedEvent[] {
    this.ended = true;
    return this.events;
  }
}

// ─── State ─────────────────────────────────────────────────────────────────────

let appMode: AppMode = "enroll";
let enrollmentT0 = 0;
let sessionActive = false;
let attempts: AttemptRecord[] = [];
let currentCapture: AttemptCapture | null = null;
let targetPhrase = CONFIG.targetPhrase;
let requiredCount = CONFIG.requiredAttempts;
let devicePreset: DevicePreset = CONFIG.defaultDevicePreset;

// ─── DOM refs ──────────────────────────────────────────────────────────────────

const viewSetup = $("view-setup") as HTMLElement;
const viewTyping = $("view-typing") as HTMLElement;
const viewResult = $("view-result") as HTMLElement;
const elUserId = $("userId") as HTMLInputElement;
const elPrompt = $("prompt") as HTMLElement;
const elTyping = $("typing") as HTMLTextAreaElement;
const elProgressFill = $("progressFill") as HTMLElement;
const elProgressHint = $("progressHint") as HTMLElement;
const elStatusSetup = $("statusSetup") as HTMLElement;
const elStatus = $("status") as HTMLElement;
const elJsonPreview = $("jsonPreview") as HTMLElement;
const btnStart = $("btnStart") as HTMLButtonElement;
const btnBack = $("btnBack") as HTMLButtonElement;
const btnReset = $("btnReset") as HTMLButtonElement;
const btnResultBack = $("btnResultBack") as HTMLButtonElement;
const deviceBtns = viewSetup.querySelectorAll<HTMLButtonElement>(".device-btn");
const modeTabs = viewSetup.querySelectorAll<HTMLButtonElement>(".mode-tab");
const progressField = $("progressField") as HTMLElement;
const typingHint = $("typingHint") as HTMLElement;
const typingTitle = $("typingTitle") as HTMLElement;
const elResultBadge = $("resultBadge") as HTMLElement;
const elResultMessage = $("resultMessage") as HTMLElement;
const elResultDetails = $("resultDetails") as HTMLElement;

// ─── UI ────────────────────────────────────────────────────────────────────────

function applyDeviceUi(): void {
  elTyping.style.fontSize = devicePreset === "mobile" ? "16px" : "1rem";
}

function setDevicePreset(p: DevicePreset): void {
  devicePreset = p;
  deviceBtns.forEach((b) => {
    b.classList.toggle("active", b.dataset.device === p);
  });
  applyDeviceUi();
}

function setMode(m: AppMode): void {
  appMode = m;
  modeTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === m);
    tab.setAttribute("aria-selected", String(tab.dataset.mode === m));
  });
  btnStart.textContent = m === "enroll" ? "Начать регистрацию" : "Начать верификацию";
}

function setSetupLocked(locked: boolean): void {
  elUserId.disabled = locked;
  deviceBtns.forEach((b) => { b.disabled = locked; });
  (modeTabs as NodeListOf<HTMLButtonElement>).forEach((t) => { t.disabled = locked; });
  btnStart.disabled = locked;
}

function setResetVisible(visible: boolean): void {
  btnReset.hidden = !visible;
}

function showSetupView(): void {
  viewSetup.hidden = false;
  viewTyping.hidden = true;
  viewResult.hidden = true;
}

function showTypingView(): void {
  viewSetup.hidden = true;
  viewTyping.hidden = false;
  viewResult.hidden = true;
}

function showResultView(data: {
  accepted: boolean;
  score: number;
  threshold: number;
  confidence: number;
  message: string;
}): void {
  viewSetup.hidden = true;
  viewTyping.hidden = true;
  viewResult.hidden = false;

  elResultBadge.textContent = data.accepted ? "✓ Доступ разрешён" : "✗ Доступ запрещён";
  elResultBadge.className = `result-badge ${data.accepted ? "accepted" : "rejected"}`;
  elResultMessage.textContent = data.message;
  elResultDetails.innerHTML = `
    <div class="result-row"><span>Уверенность</span><strong>${Math.round(data.confidence * 100)}%</strong></div>
    <div class="result-row"><span>Оценка</span><strong>${data.score.toFixed(3)}</strong></div>
    <div class="result-row"><span>Порог</span><strong>${data.threshold.toFixed(3)}</strong></div>
  `;
}

function updateProgress(): void {
  const n = attempts.length;
  const pct = requiredCount > 0 ? Math.min(100, (n / requiredCount) * 100) : 0;
  elProgressFill.style.width = `${pct}%`;
  if (!sessionActive && n !== 0) {
    elProgressHint.textContent = `${n}/${requiredCount}. Регистрация прошла успешно.`;
  } else {
    elProgressHint.textContent = `${n}/${requiredCount}`;
  }
}

// ─── Payload ───────────────────────────────────────────────────────────────────

function toBackendAttempt(rec: AttemptRecord): BackendAttempt {
  return { attemptId: rec.attemptId, events: rec.events };
}

function buildEnrollPayload(): EnrollPayload {
  return {
    login: elUserId.value.trim() || CONFIG.userIdPlaceholder,
    device_type: devicePreset,
    phrase: targetPhrase,
    attempts: attempts.map(toBackendAttempt),
  };
}

function buildVerifyPayload(rec: AttemptRecord): VerifyPayload {
  return {
    login: elUserId.value.trim() || CONFIG.userIdPlaceholder,
    device_type: devicePreset,
    phrase: targetPhrase,
    attempt: toBackendAttempt(rec),
  };
}

function refreshPreview(): void {
  if (appMode === "enroll") {
    elJsonPreview.textContent = JSON.stringify(buildEnrollPayload(), null, 2);
  } else if (attempts.length > 0) {
    elJsonPreview.textContent = JSON.stringify(buildVerifyPayload(attempts[0]), null, 2);
  }
}

// ─── API ───────────────────────────────────────────────────────────────────────

function apiUrl(path: string): string | null {
  const base = CONFIG.apiBaseUrl.trim().replace(/\/$/, "");
  if (!base) return null;
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${base}${p}`;
}

async function trySendEnroll(): Promise<void> {
  const url = apiUrl(CONFIG.enrollmentPath);
  if (!url) return;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildEnrollPayload()),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({})) as { detail?: string };
      elStatusSetup.textContent = `Ошибка ${res.status}: ${err.detail ?? "неизвестная ошибка"}`;
      elStatusSetup.className = "status err";
      return;
    }
    const data = await res.json() as { message: string; attempts_count: number };
    elStatusSetup.textContent = `${data.message} (попыток: ${data.attempts_count})`;
    elStatusSetup.className = "status ok";
  } catch {
    elStatusSetup.textContent = "Не удалось отправить (сеть или CORS). Смотрите JSON ниже.";
    elStatusSetup.className = "status err";
  }
}

async function trySendVerify(rec: AttemptRecord): Promise<void> {
  const url = apiUrl(CONFIG.verifyPath);
  if (!url) {
    elStatus.textContent = "apiBaseUrl не задан — верификация недоступна.";
    elStatus.className = "status err";
    setTimeout(showSetupView, 2500);
    return;
  }
  elStatus.textContent = "Отправляю на сервер…";
  elStatus.className = "status";
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildVerifyPayload(rec)),
    });
    if (res.status === 404) {
      elStatus.textContent = "Пользователь не найден. Сначала пройдите регистрацию.";
      elStatus.className = "status err";
      setTimeout(showSetupView, 2500);
      return;
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({})) as { detail?: string };
      elStatus.textContent = `Ошибка ${res.status}: ${err.detail ?? "неизвестная ошибка"}`;
      elStatus.className = "status err";
      setTimeout(showSetupView, 2500);
      return;
    }
    const data = await res.json() as {
      accepted: boolean;
      score: number;
      threshold: number;
      confidence: number;
      message: string;
    };
    showResultView(data);
  } catch {
    elStatus.textContent = "Не удалось отправить (сеть или CORS).";
    elStatus.className = "status err";
    setTimeout(showSetupView, 2500);
  }
}

// ─── Flow ──────────────────────────────────────────────────────────────────────

function beginEnrollment(): void {
  const uid = elUserId.value.trim();
  if (!uid) {
    elStatusSetup.textContent = "Введите имя или логин.";
    elStatusSetup.className = "status err";
    return;
  }
  elStatusSetup.textContent = "";
  elStatusSetup.className = "status";

  targetPhrase = CONFIG.targetPhrase;
  requiredCount = CONFIG.requiredAttempts;
  attempts = [];
  sessionActive = true;
  enrollmentT0 = nowMs();
  currentCapture = null;
  elTyping.disabled = false;
  elTyping.value = "";
  elPrompt.textContent = targetPhrase;
  elStatus.textContent = "";
  elStatus.className = "status";
  setSetupLocked(true);
  setResetVisible(true);
  progressField.hidden = false;
  typingTitle.textContent = "Набор фразы — Регистрация";
  typingHint.textContent =
    "Наберите контрольную фразу необходимое количество раз. Набирайте естественно, не копируя. Следующая попытка автоматически переключится при полном совпадении.";
  updateProgress();
  refreshPreview();
  applyDeviceUi();
  showTypingView();
  elTyping.focus();
}

function beginVerify(): void {
  const uid = elUserId.value.trim();
  if (!uid) {
    elStatusSetup.textContent = "Введите имя или логин.";
    elStatusSetup.className = "status err";
    return;
  }
  elStatusSetup.textContent = "";
  elStatusSetup.className = "status";

  targetPhrase = CONFIG.targetPhrase;
  requiredCount = 1;
  attempts = [];
  sessionActive = true;
  enrollmentT0 = nowMs();
  currentCapture = null;
  elTyping.disabled = false;
  elTyping.value = "";
  elPrompt.textContent = targetPhrase;
  elStatus.textContent = "";
  elStatus.className = "status";
  setSetupLocked(true);
  setResetVisible(false);
  progressField.hidden = true;
  typingTitle.textContent = "Набор фразы — Верификация";
  typingHint.textContent =
    "Наберите контрольную фразу один раз. Набирайте естественно, не копируя.";
  updateProgress();
  refreshPreview();
  applyDeviceUi();
  showTypingView();
  elTyping.focus();
}

function backToSetup(): void {
  sessionActive = false;
  currentCapture = null;
  attempts = [];
  elTyping.disabled = true;
  elTyping.value = "";
  elStatus.textContent = "";
  elStatus.className = "status";
  elStatusSetup.textContent = "";
  elStatusSetup.className = "status";
  setSetupLocked(false);
  setResetVisible(false);
  updateProgress();
  refreshPreview();
  showSetupView();
  elUserId.focus();
}

function restartTypingSession(): void {
  if (!sessionActive) return;
  enrollmentT0 = nowMs();
  attempts = [];
  currentCapture = null;
  elTyping.disabled = false;
  elTyping.value = "";
  elPrompt.textContent = targetPhrase;
  elStatus.textContent = "";
  elStatus.className = "status";
  setResetVisible(true);
  updateProgress();
  refreshPreview();
  elTyping.focus();
}

function ensureCapture(): AttemptCapture | null {
  if (!sessionActive) return null;
  if (!currentCapture) {
    currentCapture = new AttemptCapture(enrollmentT0);
  }
  return currentCapture;
}

function typingSpeedCpm(charCount: number, startedAt: number, endedAt: number): number {
  const dur = endedAt - startedAt;
  if (dur <= 0) return 0;
  return Math.round((charCount * 60000) / dur);
}

function completeAttempt(finalText: string): void {
  if (!sessionActive || !currentCapture) return;
  const perfEnd = nowMs();
  const startedAt = currentCapture.startedAtSession();
  const endedAt = currentCapture.endedAtSession(perfEnd);
  const evs = currentCapture.finish();
  const attemptId = `att_${attempts.length + 1}`;
  const rec: AttemptRecord = {
    attemptId,
    startedAt,
    endedAt,
    targetText: targetPhrase,
    finalText,
    typing_speed: typingSpeedCpm(finalText.length, startedAt, endedAt),
    events: evs,
  };
  attempts.push(rec);
  currentCapture = null;

  if (appMode === "verify") {
    elTyping.disabled = true;
    sessionActive = false;
    setSetupLocked(false);
    elStatus.textContent = "Проверяю…";
    elStatus.className = "status";
    refreshPreview();
    void trySendVerify(rec);
    return;
  }

  updateProgress();
  refreshPreview();

  if (attempts.length >= requiredCount) {
    elTyping.disabled = true;
    sessionActive = false;
    setSetupLocked(false);
    setResetVisible(false);
    elStatus.textContent = "";
    elStatus.className = "status";
    elStatusSetup.textContent = "Готово. Отправляю на сервер…";
    elStatusSetup.className = "status ok";
    updateProgress();
    showSetupView();
    void trySendEnroll();
    return;
  }

  elTyping.value = "";
  elTyping.focus();
}

// ─── Обработчики событий ───────────────────────────────────────────────────────

function onFocusField(): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  cap.startAttemptWall(perf);
  cap.push({ type: "focus", t: cap.localT(perf) }, perf);
}

function onBlurField(): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  cap.push({ type: "blur", t: cap.localT(perf) }, perf);
  refreshPreview();
}

function onPaste(): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  cap.push({ type: "paste", t: cap.localT(perf) }, perf);
  refreshPreview();
}

function onKeyDown(ev: KeyboardEvent): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  const snap = caretSnapshot(elTyping);
  cap.push(
    {
      type: "keydown",
      key: ev.key,
      code: ev.code,
      location: ev.location,
      t: cap.localT(perf),
      repeat: ev.repeat,
      value: snap.value,
      caretStart: snap.caretStart,
      caretEnd: snap.caretEnd,
    },
    perf
  );
  refreshPreview();
}

function onKeyUp(ev: KeyboardEvent): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  const snap = caretSnapshot(elTyping);
  cap.push(
    {
      type: "keyup",
      key: ev.key,
      code: ev.code,
      location: ev.location,
      t: cap.localT(perf),
      repeat: ev.repeat,
      value: snap.value,
      caretStart: snap.caretStart,
      caretEnd: snap.caretEnd,
    },
    perf
  );
  refreshPreview();
}

function onBeforeInput(ev: InputEvent): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  const snap = caretSnapshot(elTyping);
  cap.push(
    {
      type: "beforeinput",
      inputType: ev.inputType,
      data: ev.data,
      t: cap.localT(perf),
      value: snap.value,
      caretStart: snap.caretStart,
      caretEnd: snap.caretEnd,
    },
    perf
  );
  refreshPreview();
}

function onInput(ev: Event): void {
  const cap = ensureCapture();
  if (!cap) return;
  const ie = ev as InputEvent;
  if (ie.isComposing) return;
  const perf = nowMs();
  const snap = caretSnapshot(elTyping);
  cap.push(
    {
      type: "input",
      inputType: ie.inputType || "",
      data: ie.data ?? null,
      value: snap.value,
      caretStart: snap.caretStart,
      caretEnd: snap.caretEnd,
      t: cap.localT(perf),
    },
    perf
  );
  refreshPreview();

  if (snap.value === targetPhrase) {
    completeAttempt(snap.value);
  }
}

function onCompositionStart(ev: CompositionEvent): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  cap.push({ type: "compositionstart", t: cap.localT(perf), data: ev.data || "" }, perf);
  refreshPreview();
}

function onCompositionUpdate(ev: CompositionEvent): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  cap.push({ type: "compositionupdate", t: cap.localT(perf), data: ev.data || "" }, perf);
  refreshPreview();
}

function onCompositionEnd(ev: CompositionEvent): void {
  const cap = ensureCapture();
  if (!cap) return;
  const perf = nowMs();
  cap.push({ type: "compositionend", t: cap.localT(perf), data: ev.data || "" }, perf);
  refreshPreview();
  queueMicrotask(() => {
    const snap = caretSnapshot(elTyping);
    if (snap.value === targetPhrase) {
      completeAttempt(snap.value);
    }
  });
}

// ─── Wire UI ───────────────────────────────────────────────────────────────────

function wireUi(): void {
  elUserId.placeholder = CONFIG.userIdPlaceholder;
  setDevicePreset(CONFIG.defaultDevicePreset);
  setMode("enroll");

  deviceBtns.forEach((b) => {
    b.addEventListener("click", () => {
      if (b.disabled) return;
      const d = b.dataset.device as DevicePreset;
      if (d === "desktop" || d === "mobile") setDevicePreset(d);
    });
  });

  modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      if ((tab as HTMLButtonElement).disabled) return;
      const m = tab.dataset.mode as AppMode;
      if (m === "enroll" || m === "verify") setMode(m);
    });
  });

  btnStart.addEventListener("click", () => {
    if (appMode === "enroll") beginEnrollment();
    else beginVerify();
  });
  btnBack.addEventListener("click", backToSetup);
  btnReset.addEventListener("click", restartTypingSession);
  btnResultBack.addEventListener("click", () => {
    showSetupView();
    setSetupLocked(false);
    elUserId.focus();
  });

  elTyping.addEventListener("focus", onFocusField);
  elTyping.addEventListener("blur", onBlurField);
  elTyping.addEventListener("paste", onPaste);
  elTyping.addEventListener("keydown", onKeyDown);
  elTyping.addEventListener("keyup", onKeyUp);
  elTyping.addEventListener("beforeinput", onBeforeInput);
  elTyping.addEventListener("input", onInput);
  elTyping.addEventListener("compositionstart", onCompositionStart);
  elTyping.addEventListener("compositionupdate", onCompositionUpdate);
  elTyping.addEventListener("compositionend", onCompositionEnd);

  elPrompt.textContent = CONFIG.targetPhrase;
  elTyping.disabled = true;
  setSetupLocked(false);
  setResetVisible(false);
  showSetupView();
  updateProgress();
  refreshPreview();
}

wireUi();
