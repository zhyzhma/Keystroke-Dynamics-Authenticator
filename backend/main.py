import json
import sys
import argparse
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

from model import KeystrokeModel, extract_training_data, extract_single_vector

NG_SEP = "␟"

def authenticate_user(parsed_json: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    model = KeystrokeModel()
    model.load(model_path)

    vector = extract_single_vector(parsed_json)
    result = model.predict(vector)

    return {
        "accepted": result["accepted"],
        "score": result["score"],
        "threshold": result["threshold"],
        "confidence": result["score"] - result["threshold"]
    }


def safe_mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "std": 0.0, "min": v, "max": v, "median": v, "count": 1}
    vals = [float(v) for v in values]
    return {
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "median": float(median(vals)),
        "count": len(vals),
    }


def normalize_code(code: Optional[str], key: Optional[str]) -> str:
    if code:
        return str(code)
    if key == " ":
        return "Space"
    if key is None:
        return "Unknown"
    return str(key)


def printable_symbol_from_event(ev: Dict[str, Any]) -> Optional[str]:
    key = ev.get("key")
    code = ev.get("code")
    if code == "Space" or key == " ":
        return " "
    if isinstance(key, str) and len(key) == 1:
        return key
    return None


def make_ngram_key(tokens: List[str]) -> str:
    return NG_SEP.join(tokens)


def word_spans(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    start = None
    for i, ch in enumerate(text):
        if not ch.isspace() and start is None:
            start = i
        elif ch.isspace() and start is not None:
            spans.append((start, i, text[start:i]))
            start = None
    if start is not None:
        spans.append((start, len(text), text[start:]))
    return spans


def word_at_pos(text: str, pos: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
    if pos is None:
        return None, None
    for idx, (s, e, w) in enumerate(word_spans(text)):
        if s <= pos <= e:
            return idx, w
    for idx, (s, e, w) in enumerate(word_spans(text)):
        if pos == e:
            return idx, w
    return None, None


def diff_ops(prev: str, curr: str) -> List[Dict[str, Any]]:
    sm = SequenceMatcher(a=prev, b=curr)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ops.append(
            {
                "tag": tag,
                "prev_start": i1,
                "prev_end": i2,
                "curr_start": j1,
                "curr_end": j2,
                "removed": prev[i1:i2],
                "inserted": curr[j1:j2],
            }
        )
    return ops


def flatten_numeric(d: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                out.update(flatten_numeric(v, key))
            elif isinstance(v, bool):
                out[key] = float(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[key] = float(v)
    return out


class AttemptExtractor:
    def __init__(self) -> None:
        self.pending_down: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self.active_mods = {
            "ShiftLeft": False,
            "ShiftRight": False,
            "ControlLeft": False,
            "ControlRight": False,
            "AltLeft": False,
            "AltRight": False,
            "CapsLock": False,
        }

        self.dwell_by_code: Dict[str, List[float]] = defaultdict(list)
        self.global_dwell: List[float] = []

        self.flight_times: List[float] = []
        self.down_down_times: List[float] = []
        self.up_up_times: List[float] = []

        self.printable_records: List[Dict[str, Any]] = []

        self.digraphs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.trigraphs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        self.total_keydowns = 0
        self.total_keyups = 0
        self.total_repeats = 0
        self.total_printable_keydowns = 0

        self.backspace_count = 0
        self.delete_count = 0
        self.paste_count = 0
        self.input_delete_count = 0
        self.input_insert_count = 0

        self.error_positions: List[Dict[str, Any]] = []
        self.error_word_counter: Counter = Counter()
        self.error_char_counter: Counter = Counter()
        self.error_sequences: Counter = Counter()

        self.current_text: str = ""
        self.prev_input_text: Optional[str] = None
        self.last_edit_kind: Optional[str] = None
        self.last_edit_time: Optional[float] = None

        self.shift_left_count = 0
        self.shift_right_count = 0
        self.ctrl_left_count = 0
        self.ctrl_right_count = 0
        self.alt_left_count = 0
        self.alt_right_count = 0
        self.capslock_toggles = 0

        self.uppercase_via_shift = 0
        self.uppercase_via_capslock = 0
        self.uppercase_other = 0
        self.lowercase_count = 0

        self.session_start_t: Optional[float] = None
        self.session_end_t: Optional[float] = None

    def _set_mod_state(self, code: str, down: bool) -> None:
        if code in self.active_mods and code != "CapsLock":
            self.active_mods[code] = down
        if code == "CapsLock" and down:
            self.active_mods["CapsLock"] = not self.active_mods["CapsLock"]
            self.capslock_toggles += 1

    def _shift_active(self) -> bool:
        return self.active_mods["ShiftLeft"] or self.active_mods["ShiftRight"]

    def _ctrl_active(self) -> bool:
        return self.active_mods["ControlLeft"] or self.active_mods["ControlRight"]

    def _alt_active(self) -> bool:
        return self.active_mods["AltLeft"] or self.active_mods["AltRight"]

    def _duration_ms(self, events: List[Dict[str, Any]]) -> float:
        started = None
        ended = None

        for ev in events:
            if ev.get("type") == "focus":
                started = float(ev.get("t", 0.0))
                break

        for ev in reversed(events):
            if ev.get("type") == "blur":
                ended = float(ev.get("t", 0.0))
                break

        if started is None and events:
            started = float(events[0].get("t", 0.0))
        if ended is None and events:
            ended = float(events[-1].get("t", 0.0))

        if started is None or ended is None:
            return 0.0
        return max(0.0, ended - started)

    def process(self, events: List[Dict[str, Any]], target_text: str = "", final_text: str = "") -> Dict[str, Any]:
        # Нормализация типов событий (поддержка обоих форматов)
        for ev in events:
            etype = ev.get("type")
            if etype == "down":
                ev["type"] = "keydown"
            elif etype == "up":
                ev["type"] = "keyup"

        events = sorted(events, key=lambda e: float(e.get("t", 0.0)))

        if events:
            self.session_start_t = float(events[0].get("t", 0.0))
            self.session_end_t = float(events[-1].get("t", 0.0))

        for ev in events:
            etype = ev.get("type")
            t = float(ev.get("t", 0.0))
            key = ev.get("key")
            code = normalize_code(ev.get("code"), key)
            repeat = bool(ev.get("repeat", False))

            if etype in ("focus", "blur", "paste", "compositionstart", "compositionupdate", "compositionend"):
                if etype == "paste":
                    self.paste_count += 1
                continue

            if etype == "keydown":
                self.total_keydowns += 1
                if repeat:
                    self.total_repeats += 1

                if key in ("Backspace", "Delete") or code in ("Backspace", "Delete"):
                    if key == "Backspace" or code == "Backspace":
                        self.backspace_count += 1
                    else:
                        self.delete_count += 1

                if code in ("ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight", "CapsLock"):
                    if code == "ShiftLeft":
                        self.shift_left_count += 1
                    elif code == "ShiftRight":
                        self.shift_right_count += 1
                    elif code == "ControlLeft":
                        self.ctrl_left_count += 1
                    elif code == "ControlRight":
                        self.ctrl_right_count += 1
                    elif code == "AltLeft":
                        self.alt_left_count += 1
                    elif code == "AltRight":
                        self.alt_right_count += 1
                    self._set_mod_state(code, True)
                else:
                    sym = printable_symbol_from_event(ev)
                    if sym is not None:
                        if sym.isalpha():
                            if sym.isupper():
                                if self._shift_active() and not self.active_mods["CapsLock"]:
                                    self.uppercase_via_shift += 1
                                elif self.active_mods["CapsLock"] and not self._shift_active():
                                    self.uppercase_via_capslock += 1
                                else:
                                    self.uppercase_other += 1
                            else:
                                self.lowercase_count += 1

                        if not repeat:
                            self.total_printable_keydowns += 1
                            self.printable_records.append(
                                {
                                    "symbol": sym,
                                    "code": code,
                                    "down_t": t,
                                    "up_t": None,
                                    "repeat": repeat,
                                }
                            )

                self.pending_down[code].append({"t": t, "ev": ev})

            elif etype == "keyup":
                self.total_keyups += 1
                if code in ("ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight"):
                    self._set_mod_state(code, False)

                if self.pending_down[code]:
                    down_item = self.pending_down[code].pop()
                    dwell = t - float(down_item["t"])
                    self.dwell_by_code[code].append(dwell)
                    self.global_dwell.append(dwell)

                # Обновляем up_t для printable символов
                if self.printable_records:
                    for rec in reversed(self.printable_records):
                        if rec["code"] == code and rec["up_t"] is None:
                            rec["up_t"] = t
                            break

            elif etype == "input":
                input_type = ev.get("inputType", "") or ""
                value = ev.get("value")

                if input_type.startswith("delete"):
                    self.input_delete_count += 1
                    if self.last_edit_kind in ("insert_input", "insert"):
                        self.error_sequences["insert->delete"] += 1
                elif input_type.startswith("insert"):
                    self.input_insert_count += 1

                if value is not None:
                    new_text = str(value)
                    prev_text = self.current_text

                    if new_text != prev_text:
                        ops = diff_ops(prev_text, new_text)
                        for op in ops:
                            pos = int(op["prev_start"])
                            removed = op["removed"]
                            inserted = op["inserted"]
                            op_tag = op["tag"]

                            if op_tag in ("delete", "replace"):
                                wi, word = word_at_pos(prev_text, pos)
                                self.error_positions.append(
                                    {
                                        "time": t,
                                        "op": op_tag,
                                        "position": pos,
                                        "word_index": wi,
                                        "word": word,
                                        "removed": removed,
                                        "inserted": inserted,
                                        "caretStart": ev.get("caretStart"),
                                        "caretEnd": ev.get("caretEnd"),
                                    }
                                )
                                if word is not None:
                                    self.error_word_counter[word] += 1
                                for ch in removed:
                                    self.error_char_counter[ch] += 1

                            if op_tag in ("insert", "replace"):
                                if self.last_edit_kind in ("delete", "backspace", "delete_input"):
                                    self.error_sequences["delete->insert"] += 1

                        self.prev_input_text = prev_text
                        self.current_text = new_text

                self.current_text = new_text

                if input_type.startswith("delete"):
                    self.last_edit_kind = "delete_input"
                    self.last_edit_time = t
                elif input_type.startswith("insert"):
                    self.last_edit_kind = "insert_input"
                    self.last_edit_time = t

        # === Пересчёт таймингов только по printable символам ===
        self.flight_times = []
        self.down_down_times = []
        self.up_up_times = []

        seq = [r for r in self.printable_records if r.get("down_t") is not None]
        seq.sort(key=lambda r: float(r["down_t"]))

        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            sym_a, sym_b = a["symbol"], b["symbol"]
            if sym_a is None or sym_b is None:
                continue

            key2 = make_ngram_key([sym_a, sym_b])
            dd = float(b["down_t"]) - float(a["down_t"])
            self.down_down_times.append(dd)
            self.digraphs[key2]["down_down_ms"].append(dd)

            if a.get("up_t") is not None:
                flight = float(b["down_t"]) - float(a["up_t"])
                self.flight_times.append(flight)
                self.digraphs[key2]["flight_ms"].append(flight)

        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a.get("up_t") is not None and b.get("up_t") is not None:
                self.up_up_times.append(float(b["up_t"]) - float(a["up_t"]))

        # Trigraphs
        for i in range(len(seq) - 2):
            a, b, c = seq[i], seq[i + 1], seq[i + 2]
            sym_a, sym_b, sym_c = a["symbol"], b["symbol"], c["symbol"]
            if None in (sym_a, sym_b, sym_c):
                continue
            key3 = make_ngram_key([sym_a, sym_b, sym_c])
            span = float(c["down_t"]) - float(a["down_t"])
            self.trigraphs[key3]["span_ms"].append(span)

        # === Финальные метрики ===
        target = target_text or ""
        final = final_text or self.current_text or target
        duration_ms = self._duration_ms(events)
        char_count = len(final)
        typing_speed_cpm = (char_count / duration_ms * 60000.0) if duration_ms > 0 else 0.0

        per_key_stats = {}
        for code, vals in sorted(self.dwell_by_code.items(), key=lambda kv: kv[0]):
            s = safe_mean_std(vals)
            per_key_stats[code] = {
                "count": s["count"],
                "mean_dwell_ms": s["mean"],
                "std_dwell_ms": s["std"],
                "min_dwell_ms": s["min"],
                "max_dwell_ms": s["max"],
                "median_dwell_ms": s["median"],
            }

        dwell_summary = safe_mean_std(self.global_dwell)
        flight_summary = safe_mean_std(self.flight_times)
        dd_summary = safe_mean_std(self.down_down_times)
        uu_summary = safe_mean_std(self.up_up_times)

        digraph_stats = {
            k: {
                "count": max(len(v.get("down_down_ms", [])), len(v.get("flight_ms", []))),
                "down_down_ms": safe_mean_std(v.get("down_down_ms", [])),
                "flight_ms": safe_mean_std(v.get("flight_ms", [])),
            }
            for k, v in sorted(self.digraphs.items(), key=lambda kv: kv[0])
        }

        trigraph_stats = {
            k: {
                "count": len(v.get("span_ms", [])),
                "span_ms": safe_mean_std(v.get("span_ms", [])),
            }
            for k, v in sorted(self.trigraphs.items(), key=lambda kv: kv[0])
        }

        total_errors_keydown = self.backspace_count + self.delete_count
        total_keystrokes = max(self.total_keydowns, 1)
        error_rate = total_errors_keydown / total_keystrokes

        errors = {
            "backspace_count": self.backspace_count,
            "delete_count": self.delete_count,
            "paste_count": self.paste_count,
            "input_delete_count": self.input_delete_count,
            "input_insert_count": self.input_insert_count,
            "error_rate": error_rate,
            "error_words": dict(self.error_word_counter),
            "error_chars": dict(self.error_char_counter),
            "error_positions": self.error_positions,
            "correction_sequences": dict(self.error_sequences),
            "total_errors_keydown": total_errors_keydown,
        }

        modifiers = {
            "left_shift_count": self.shift_left_count,
            "right_shift_count": self.shift_right_count,
            "left_ctrl_count": self.ctrl_left_count,
            "right_ctrl_count": self.ctrl_right_count,
            "left_alt_count": self.alt_left_count,
            "right_alt_count": self.alt_right_count,
            "capslock_toggles": self.capslock_toggles,
            "uppercase_via_shift": self.uppercase_via_shift,
            "uppercase_via_capslock": self.uppercase_via_capslock,
            "uppercase_other": self.uppercase_other,
            "lowercase_count": self.lowercase_count,
            "left_shift_ratio": (
                self.shift_left_count / (self.shift_left_count + self.shift_right_count)
                if (self.shift_left_count + self.shift_right_count) > 0 else 0.0
            ),
            "left_ctrl_ratio": (
                self.ctrl_left_count / (self.ctrl_left_count + self.ctrl_right_count)
                if (self.ctrl_left_count + self.ctrl_right_count) > 0 else 0.0
            ),
        }

        attempt_features = {
            "meta": {
                "targetText": target,
                "finalText": final,
                "durationMs": duration_ms,
                "charCount": char_count,
                "totalKeydowns": self.total_keydowns,
                "totalKeyups": self.total_keyups,
                "totalRepeats": self.total_repeats,
                "printableKeydowns": self.total_printable_keydowns,
            },
            "timings": {
                "dwell": dwell_summary,
                "flight": flight_summary,
                "downDown": dd_summary,
                "upUp": uu_summary,
            },
            "per_key_stats": per_key_stats,
            "digraphs": digraph_stats,
            "trigraphs": trigraph_stats,
            "typing": {
                "typing_speed_cpm": typing_speed_cpm,
                "typing_speed_wpm": typing_speed_cpm / 5.0 if typing_speed_cpm else 0.0,
            },
            "errors": errors,
            "modifiers": modifiers,
        }

        flat = flatten_numeric(attempt_features)
        feature_names = sorted(flat.keys())
        feature_vector = [flat[k] for k in feature_names]

        attempt_features["flat_features"] = flat
        attempt_features["feature_names"] = feature_names
        attempt_features["feature_vector"] = feature_vector

        return attempt_features


def transform_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    attempts = payload.get("attempts", [])
    out_attempts = []

    for idx, attempt in enumerate(attempts):
        extractor = AttemptExtractor()
        target = attempt.get("targetText") or payload.get("phrase", "")
        final = attempt.get("finalText") or ""
        feats = extractor.process(attempt.get("events", []), target_text=target, final_text=final)
        out_attempts.append(
            {
                "attemptId": attempt.get("attemptId", f"attempt_{idx + 1}"),
                "features": feats,
            }
        )

    return {
        "sessionId": payload.get("sessionId"),
        "userId": payload.get("userId"),
        "phrase": payload.get("phrase"),
        "layout": payload.get("layout"),
        "attempts": out_attempts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Transform keystroke raw JSON into aggregated features.")
    ap.add_argument("input", nargs="?", help="Path to JSON input file. If omitted, read from stdin.")
    ap.add_argument("--output", "-o", help="Path to write JSON output. If omitted, print to stdout.")
    args = ap.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.load(sys.stdin)

    result = transform_payload(payload)
    text = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        print(text)


    # === Train ===    
    vectors, feature_names = extract_training_data(result)

    model = KeystrokeModel()
    model.fit(vectors, feature_names)

    # сохраняем
    model.save("keystroke_model.pkl")

    print("Model trained and saved.")


if __name__ == "__main__":
    main()