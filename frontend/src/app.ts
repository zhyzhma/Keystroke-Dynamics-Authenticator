/**
 * KDA enrollment: сбор событий клавиатуры и сборка JSON для бэкенда.
 *
 * В интерфейсе: имя пользователя и Desktop/Mobile.
 * Здесь (для разработчика): фраза, число попыток, дефолтный placeholder имени, URL API.
 */

type DevicePreset = "desktop" | "mobile";

const CONFIG = {
  /** Подсказка в поле «Имя», если оно пустое. */
  userIdPlaceholder: "local-user",
  targetPhrase: "The quick brown fox jumps over the lazy dog",
  requiredAttempts: 35,
  defaultDevicePreset: "desktop" as DevicePreset,
  /**
   * После сбора всех попыток отправить POST с JSON на бэкенд.
   * Пример: "http://localhost:8080" или "https://api.example.com"
   * Пустая строка — не отправлять, только превью в <details>.
   */
  apiBaseUrl: "",
  /** Путь относительно apiBaseUrl, без хвостового слэша у base. */
  enrollmentPath: "/api/enroll",
};

interface EnrollmentPayload {
  userId: string;
  devicePreset: DevicePreset;
  phrase: string;
  attempts: AttemptRecord[];
}

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

let enrollmentT0 = 0;
let sessionActive = false;
let attempts: AttemptRecord[] = [];
let currentCapture: AttemptCapture | null = null;
let targetPhrase = CONFIG.targetPhrase;
let requiredCount = CONFIG.requiredAttempts;
let devicePreset: DevicePreset = CONFIG.defaultDevicePreset;

const viewSetup = $("view-setup") as HTMLElement;
const viewTyping = $("view-typing") as HTMLElement;
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
const deviceBtns = viewSetup.querySelectorAll<HTMLButtonElement>(".device-btn");

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

function setSetupLocked(locked: boolean): void {
  elUserId.disabled = locked;
  deviceBtns.forEach((b) => {
    b.disabled = locked;
  });
  btnStart.disabled = locked;
}

function setResetVisible(visible: boolean): void {
  btnReset.hidden = !visible;
}

function showSetupView(): void {
  viewSetup.hidden = false;
  viewTyping.hidden = true;
}

function showTypingView(): void {
  viewSetup.hidden = true;
  viewTyping.hidden = false;
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

function buildPayload(): EnrollmentPayload {
  return {
    userId: elUserId.value.trim() || CONFIG.userIdPlaceholder,
    devicePreset,
    phrase: targetPhrase,
    attempts,
  };
}

function refreshPreview(): void {
  elJsonPreview.textContent = JSON.stringify(buildPayload(), null, 2);
}

function typingSpeedCpm(charCount: number, startedAt: number, endedAt: number): number {
  const dur = endedAt - startedAt;
  if (dur <= 0) return 0;
  return Math.round((charCount * 60000) / dur);
}

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
  updateProgress();
  refreshPreview();
  applyDeviceUi();
  showTypingView();
  elTyping.focus();
}

/** Вернуться на экран имени и режима; текущая регистрация отменяется. */
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

/** Очистить попытки и начать набор заново (тот же пользователь и режим). */
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

async function trySendToBackend(): Promise<void> {
  const base = CONFIG.apiBaseUrl.trim().replace(/\/$/, "");
  if (!base) return;
  const path = CONFIG.enrollmentPath.startsWith("/")
    ? CONFIG.enrollmentPath
    : `/${CONFIG.enrollmentPath}`;
  const url = `${base}${path}`;
  const body = JSON.stringify(buildPayload());
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    if (!res.ok) {
      elStatusSetup.textContent = `Сервер ответил ${res.status}. Данные в JSON ниже.`;
      elStatusSetup.className = "status err";
      return;
    }
    elStatusSetup.textContent = "Отправлено на сервер.";
    elStatusSetup.className = "status ok";
  } catch {
    elStatusSetup.textContent = "Не удалось отправить (сеть или CORS). Смотрите JSON ниже.";
    elStatusSetup.className = "status err";
  }
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
  updateProgress();
  refreshPreview();

  if (attempts.length >= requiredCount) {
    elTyping.disabled = true;
    sessionActive = false;
    setSetupLocked(false);
    setResetVisible(false);
    elStatus.textContent = "";
    elStatus.className = "status";
    elStatusSetup.textContent = "Готово.";
    elStatusSetup.className = "status ok";
    updateProgress();
    showSetupView();
    void trySendToBackend();
    return;
  }

  elTyping.value = "";
  elTyping.focus();
}

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

function wireUi(): void {
  elUserId.placeholder = CONFIG.userIdPlaceholder;
  setDevicePreset(CONFIG.defaultDevicePreset);
  deviceBtns.forEach((b) => {
    b.addEventListener("click", () => {
      if (b.disabled) return;
      const d = b.dataset.device as DevicePreset;
      if (d === "desktop" || d === "mobile") setDevicePreset(d);
    });
  });

  btnStart.addEventListener("click", beginEnrollment);
  btnBack.addEventListener("click", backToSetup);
  btnReset.addEventListener("click", restartTypingSession);

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
