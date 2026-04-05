"use strict";
/**
 * KDA enrollment: сбор событий клавиатуры и сборка JSON для бэкенда.
 *
 * В интерфейсе: имя пользователя и Desktop/Mobile.
 * Здесь (для разработчика): фраза, число попыток, дефолтный placeholder имени, URL API.
 */
const CONFIG = {
    /** Подсказка в поле «Имя», если оно пустое. */
    userIdPlaceholder: "local-user",
    targetPhrase: "The quick brown fox jumps over the lazy dog",
    requiredAttempts: 35,
    defaultDevicePreset: "desktop",
    /**
     * После сбора всех попыток отправить POST с JSON на бэкенд.
     * Пример: "http://localhost:8080" или "https://api.example.com"
     * Пустая строка — не отправлять, только превью в <details>.
     */
    apiBaseUrl: "",
    /** Путь относительно apiBaseUrl, без хвостового слэша у base. */
    enrollmentPath: "/api/enroll",
};
function $(id) {
    const el = document.getElementById(id);
    if (!el)
        throw new Error(`Missing #${id}`);
    return el;
}
function caretSnapshot(el) {
    return {
        value: el.value,
        caretStart: el.selectionStart ?? 0,
        caretEnd: el.selectionEnd ?? 0,
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
let enrollmentT0 = 0;
let sessionActive = false;
let attempts = [];
let currentCapture = null;
let targetPhrase = CONFIG.targetPhrase;
let requiredCount = CONFIG.requiredAttempts;
let devicePreset = CONFIG.defaultDevicePreset;
const viewSetup = $("view-setup");
const viewTyping = $("view-typing");
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
const deviceBtns = viewSetup.querySelectorAll(".device-btn");
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
function setSetupLocked(locked) {
    elUserId.disabled = locked;
    deviceBtns.forEach((b) => {
        b.disabled = locked;
    });
    btnStart.disabled = locked;
}
function setResetVisible(visible) {
    btnReset.hidden = !visible;
}
function showSetupView() {
    viewSetup.hidden = false;
    viewTyping.hidden = true;
}
function showTypingView() {
    viewSetup.hidden = true;
    viewTyping.hidden = false;
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
function buildPayload() {
    return {
        userId: elUserId.value.trim() || CONFIG.userIdPlaceholder,
        devicePreset,
        phrase: targetPhrase,
        attempts,
    };
}
function refreshPreview() {
    elJsonPreview.textContent = JSON.stringify(buildPayload(), null, 2);
}
function typingSpeedCpm(charCount, startedAt, endedAt) {
    const dur = endedAt - startedAt;
    if (dur <= 0)
        return 0;
    return Math.round((charCount * 60000) / dur);
}
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
    updateProgress();
    refreshPreview();
    applyDeviceUi();
    showTypingView();
    elTyping.focus();
}
/** Вернуться на экран имени и режима; текущая регистрация отменяется. */
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
/** Очистить попытки и начать набор заново (тот же пользователь и режим). */
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
async function trySendToBackend() {
    const base = CONFIG.apiBaseUrl.trim().replace(/\/$/, "");
    if (!base)
        return;
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
    }
    catch {
        elStatusSetup.textContent = "Не удалось отправить (сеть или CORS). Смотрите JSON ниже.";
        elStatusSetup.className = "status err";
    }
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
        data: ie.data ?? null,
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
function wireUi() {
    elUserId.placeholder = CONFIG.userIdPlaceholder;
    setDevicePreset(CONFIG.defaultDevicePreset);
    deviceBtns.forEach((b) => {
        b.addEventListener("click", () => {
            if (b.disabled)
                return;
            const d = b.dataset.device;
            if (d === "desktop" || d === "mobile")
                setDevicePreset(d);
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
