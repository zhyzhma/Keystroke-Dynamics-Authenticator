"use strict";
/**
 * KDA: сбор событий клавиатуры для регистрации и верификации по почерку.
 */
const CONFIG = {
    userIdPlaceholder: "local-user",
    targetPhrase: "The quick brown fox jumps over the lazy dog",
    requiredAttempts: 30,
    defaultDevicePreset: "desktop",
    /**
     * URL бэкенда. Пример: "http://localhost:8000"
     * Пустая строка — не отправлять, только превью в <details>.
     */
    apiBaseUrl: "http://localhost:8000",
    enrollmentPath: "/security/enroll",
    verifyPath: "/security/verify",
};
function $(id) {
    const el = document.getElementById(id);
    if (!el)
        throw new Error(`Missing #${id}`);
    return el;
}
function caretSnapshot(el) {
    var _a, _b;
    return {
        value: el.value,
        caretStart: (_a = el.selectionStart) !== null && _a !== void 0 ? _a : 0,
        caretEnd: (_b = el.selectionEnd) !== null && _b !== void 0 ? _b : 0,
    };
}
function nowMs() {
    return performance.now();
}
class AttemptCapture {
    constructor(enrollmentT0) {
        this.attemptPerf0 = null;
        this.events = [];
        this.ended = false;
        this.enrollmentT0 = enrollmentT0;
    }
    startAttemptWall(perfNow) {
        if (this.attemptPerf0 !== null)
            return;
        this.attemptPerf0 = perfNow;
    }
    localT(perfNow) {
        if (this.attemptPerf0 === null)
            return 0;
        return Math.round((perfNow - this.attemptPerf0) * 1000) / 1000;
    }
    startedAtSession() {
        if (this.attemptPerf0 === null)
            return 0;
        return Math.round((this.attemptPerf0 - this.enrollmentT0) * 1000) / 1000;
    }
    endedAtSession(perfNow) {
        return Math.round((perfNow - this.enrollmentT0) * 1000) / 1000;
    }
    push(e, perfNow) {
        if (this.ended)
            return;
        if (this.attemptPerf0 === null && e.type !== "focus") {
            this.startAttemptWall(perfNow);
        }
        this.events.push(e);
    }
    finish() {
        this.ended = true;
        return this.events;
    }
}
// ─── State ─────────────────────────────────────────────────────────────────────
let appMode = "enroll";
let enrollmentT0 = 0;
let sessionActive = false;
let attempts = [];
let currentCapture = null;
let targetPhrase = CONFIG.targetPhrase;
let requiredCount = CONFIG.requiredAttempts;
let devicePreset = CONFIG.defaultDevicePreset;
// ─── DOM refs ──────────────────────────────────────────────────────────────────
const viewSetup = $("view-setup");
const viewTyping = $("view-typing");
const viewResult = $("view-result");
const elUserId = $("userId");
const elPrompt = $("prompt");
const elTyping = $("typing");
const elProgressFill = $("progressFill");
const elProgressHint = $("progressHint");
const elStatusSetup = $("statusSetup");
const elStatus = $("status");
const elJsonPreview = $("jsonPreview");
const btnStart = $("btnStart");
const btnBack = $("btnBack");
const btnReset = $("btnReset");
const btnResultBack = $("btnResultBack");
const deviceBtns = viewSetup.querySelectorAll(".device-btn");
const modeTabs = viewSetup.querySelectorAll(".mode-tab");
const progressField = $("progressField");
const typingHint = $("typingHint");
const typingTitle = $("typingTitle");
const elResultBadge = $("resultBadge");
const elResultMessage = $("resultMessage");
const elResultDetails = $("resultDetails");
// ─── UI ────────────────────────────────────────────────────────────────────────
function applyDeviceUi() {
    elTyping.style.fontSize = devicePreset === "mobile" ? "16px" : "1rem";
}
function setDevicePreset(p) {
    devicePreset = p;
    deviceBtns.forEach((b) => {
        b.classList.toggle("active", b.dataset.device === p);
    });
    applyDeviceUi();
}
function setMode(m) {
    appMode = m;
    modeTabs.forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.mode === m);
        tab.setAttribute("aria-selected", String(tab.dataset.mode === m));
    });
    btnStart.textContent = m === "enroll" ? "Начать регистрацию" : "Начать верификацию";
}
function setSetupLocked(locked) {
    elUserId.disabled = locked;
    deviceBtns.forEach((b) => { b.disabled = locked; });
    modeTabs.forEach((t) => { t.disabled = locked; });
    btnStart.disabled = locked;
}
function setResetVisible(visible) {
    btnReset.hidden = !visible;
}
function showSetupView() {
    viewSetup.hidden = false;
    viewTyping.hidden = true;
    viewResult.hidden = true;
}
function showTypingView() {
    viewSetup.hidden = true;
    viewTyping.hidden = false;
    viewResult.hidden = true;
}
function showResultView(data) {
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
function updateProgress() {
    const n = attempts.length;
    const pct = requiredCount > 0 ? Math.min(100, (n / requiredCount) * 100) : 0;
    elProgressFill.style.width = `${pct}%`;
    if (!sessionActive && n !== 0) {
        elProgressHint.textContent = `${n}/${requiredCount}. Регистрация прошла успешно.`;
    }
    else {
        elProgressHint.textContent = `${n}/${requiredCount}`;
    }
}
// ─── Payload ───────────────────────────────────────────────────────────────────
function toBackendAttempt(rec) {
    return { attemptId: rec.attemptId, events: rec.events };
}
function buildEnrollPayload() {
    return {
        login: elUserId.value.trim() || CONFIG.userIdPlaceholder,
        device_type: devicePreset,
        phrase: targetPhrase,
        attempts: attempts.map(toBackendAttempt),
    };
}
function buildVerifyPayload(rec) {
    return {
        login: elUserId.value.trim() || CONFIG.userIdPlaceholder,
        device_type: devicePreset,
        phrase: targetPhrase,
        attempt: toBackendAttempt(rec),
    };
}
function refreshPreview() {
    if (appMode === "enroll") {
        elJsonPreview.textContent = JSON.stringify(buildEnrollPayload(), null, 2);
    }
    else if (attempts.length > 0) {
        elJsonPreview.textContent = JSON.stringify(buildVerifyPayload(attempts[0]), null, 2);
    }
}
// ─── API ───────────────────────────────────────────────────────────────────────
function apiUrl(path) {
    const base = CONFIG.apiBaseUrl.trim().replace(/\/$/, "");
    if (!base)
        return null;
    const p = path.startsWith("/") ? path : `/${path}`;
    return `${base}${p}`;
}
async function trySendEnroll() {
    const url = apiUrl(CONFIG.enrollmentPath);
    if (!url)
        return;
    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(buildEnrollPayload()),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            elStatusSetup.textContent = `Ошибка ${res.status}: ${err.detail ?? "неизвестная ошибка"}`;
            elStatusSetup.className = "status err";
            return;
        }
        const data = await res.json();
        elStatusSetup.textContent = `${data.message} (попыток: ${data.attempts_count})`;
        elStatusSetup.className = "status ok";
    }
    catch {
        elStatusSetup.textContent = "Не удалось отправить (сеть или CORS). Смотрите JSON ниже.";
        elStatusSetup.className = "status err";
    }
}
async function trySendVerify(rec) {
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
            const err = await res.json().catch(() => ({}));
            elStatus.textContent = `Ошибка ${res.status}: ${err.detail ?? "неизвестная ошибка"}`;
            elStatus.className = "status err";
            setTimeout(showSetupView, 2500);
            return;
        }
        const data = await res.json();
        showResultView(data);
    }
    catch {
        elStatus.textContent = "Не удалось отправить (сеть или CORS).";
        elStatus.className = "status err";
        setTimeout(showSetupView, 2500);
    }
}
// ─── Flow ──────────────────────────────────────────────────────────────────────
function beginEnrollment() {
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
function beginVerify() {
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
function backToSetup() {
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
function restartTypingSession() {
    if (!sessionActive)
        return;
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
function ensureCapture() {
    if (!sessionActive)
        return null;
    if (!currentCapture) {
        currentCapture = new AttemptCapture(enrollmentT0);
    }
    return currentCapture;
}
function typingSpeedCpm(charCount, startedAt, endedAt) {
    const dur = endedAt - startedAt;
    if (dur <= 0)
        return 0;
    return Math.round((charCount * 60000) / dur);
}
function completeAttempt(finalText) {
    if (!sessionActive || !currentCapture)
        return;
    const perfEnd = nowMs();
    const startedAt = currentCapture.startedAtSession();
    const endedAt = currentCapture.endedAtSession(perfEnd);
    const evs = currentCapture.finish();
    const attemptId = `att_${attempts.length + 1}`;
    const rec = {
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
function onFocusField() {
    const cap = ensureCapture();
    if (!cap)
        return;
    const perf = nowMs();
    cap.startAttemptWall(perf);
    cap.push({ type: "focus", t: cap.localT(perf) }, perf);
}
function onBlurField() {
    const cap = ensureCapture();
    if (!cap)
        return;
    const perf = nowMs();
    cap.push({ type: "blur", t: cap.localT(perf) }, perf);
    refreshPreview();
}
function onPaste() {
    const cap = ensureCapture();
    if (!cap)
        return;
    const perf = nowMs();
    cap.push({ type: "paste", t: cap.localT(perf) }, perf);
    refreshPreview();
}
function onKeyDown(ev) {
    const cap = ensureCapture();
    if (!cap)
        return;
    const perf = nowMs();
    const snap = caretSnapshot(elTyping);
    cap.push({
        type: "keydown",
        key: ev.key,
        code: ev.code,
        location: ev.location,
        t: cap.localT(perf),
        repeat: ev.repeat,
        value: snap.value,
        caretStart: snap.caretStart,
        caretEnd: snap.caretEnd,
    }, perf);
    refreshPreview();
}
function onKeyUp(ev) {
    const cap = ensureCapture();
    if (!cap)
        return;
    const perf = nowMs();
    const snap = caretSnapshot(elTyping);
    cap.push({
        type: "keyup",
        key: ev.key,
        code: ev.code,
        location: ev.location,
        t: cap.localT(perf),
        repeat: ev.repeat,
        value: snap.value,
        caretStart: snap.caretStart,
        caretEnd: snap.caretEnd,
    }, perf);
    refreshPreview();
}
function onBeforeInput(ev) {
    const cap = ensureCapture();
    if (!cap)
        return;
    const perf = nowMs();
    const snap = caretSnapshot(elTyping);
    cap.push({
        type: "beforeinput",
        inputType: ev.inputType,
        data: ev.data,
        t: cap.localT(perf),
        value: snap.value,
        caretStart: snap.caretStart,
        caretEnd: snap.caretEnd,
    }, perf);
    refreshPreview();
}
function onInput(ev) {
    const cap = ensureCapture();
    if (!cap)
        return;
    const ie = ev;
    if (ie.isComposing)
        return;
    const perf = nowMs();
    const snap = caretSnapshot(elTyping);
    cap.push({
        type: "input",
        inputType: ie.inputType || "",
        data: ie.data !== null && ie.data !== void 0 ? ie.data : null,
        value: snap.value,
        caretStart: snap.caretStart,
        caretEnd: snap.caretEnd,
        t: cap.localT(perf),
    }, perf);
    refreshPreview();
    if (snap.value === targetPhrase) {
        completeAttempt(snap.value);
    }
}
function onCompositionStart(ev) {
    const cap = ensureCapture();
    if (!cap)
        return;
    const perf = nowMs();
    cap.push({ type: "compositionstart", t: cap.localT(perf), data: ev.data || "" }, perf);
    refreshPreview();
}
function onCompositionUpdate(ev) {
    const cap = ensureCapture();
    if (!cap)
        return;
    const perf = nowMs();
    cap.push({ type: "compositionupdate", t: cap.localT(perf), data: ev.data || "" }, perf);
    refreshPreview();
}
function onCompositionEnd(ev) {
    const cap = ensureCapture();
    if (!cap)
        return;
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
function wireUi() {
    elUserId.placeholder = CONFIG.userIdPlaceholder;
    setDevicePreset(CONFIG.defaultDevicePreset);
    setMode("enroll");
    deviceBtns.forEach((b) => {
        b.addEventListener("click", () => {
            if (b.disabled)
                return;
            const d = b.dataset.device;
            if (d === "desktop" || d === "mobile")
                setDevicePreset(d);
        });
    });
    modeTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            if (tab.disabled)
                return;
            const m = tab.dataset.mode;
            if (m === "enroll" || m === "verify")
                setMode(m);
        });
    });
    btnStart.addEventListener("click", () => {
        if (appMode === "enroll")
            beginEnrollment();
        else
            beginVerify();
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
