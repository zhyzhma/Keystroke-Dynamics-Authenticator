import json
import sys
import argparse
from collections import defaultdict
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional

from app.ml_model.model import KeystrokeModel, extract_training_data

NG_SEP = "␟"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "std": 0.0, "min": v, "max": v, "median": v, "count": 1}
    vals = [float(v) for v in values]
    return {
        "mean":   float(mean(vals)),
        "std":    float(stdev(vals)),
        "min":    float(min(vals)),
        "max":    float(max(vals)),
        "median": float(median(vals)),
        "count":  len(vals),
    }


def normalize_string(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def normalize_code(code: Optional[str], key: Optional[str]) -> str:
    if code:
        return normalize_string(code)
    if key == " ":
        return "space"
    if key:
        return normalize_string(key)
    return "unknown"


def printable_symbol_from_event(ev: Dict[str, Any]) -> Optional[str]:
    """
    Returns the printable character for this event (lowercased for n-gram keys),
    or None if the key is non-printable.

    NOTE: the *raw* key value (before lowercasing) is used for the isupper()
    check in the modifier-tracking section of process(), so this function must
    NOT be used there — use ev.get('key') directly for that check.
    """
    key  = ev.get("key")  or ev.get("Key")
    code = ev.get("code") or ev.get("Code")
    norm_code = normalize_code(code, key)
    if norm_code == "space" or key == " ":
        return " "
    if isinstance(key, str) and len(key) == 1:
        return key.lower()   # lowercase for consistent n-gram keys
    return None


def make_ngram_key(tokens: List[str]) -> str:
    return NG_SEP.join(tokens)


def flatten_numeric(d: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            clean_k = k.lower()
            key = clean_k if not prefix else f"{prefix}.{clean_k}"
            if isinstance(v, dict):
                out.update(flatten_numeric(v, key))
            elif isinstance(v, bool):
                out[key] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                out[key] = float(v)
    return out


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class AttemptExtractor:
    """
    Extracts the full set of keystroke-dynamics features required by the spec:

    Global timing
      - dwell (hold) time          - keydown -> keyup per key
      - flight time                - keyup[i] -> keydown[i+1]
      - down-down time             - keydown[i] -> keydown[i+1]
      - up-up time                 - keyup[i]   -> keyup[i+1]

    Per-key
      - dwell stats for each physical key code

    N-grams (per pair/triple of *printable* characters)
      - digraph : flight, down-down, up-up + frequency count
      - trigraph: span (keyup[0] -> keydown[2]) + frequency count

    Errors
      - backspace count
      - delete count
      - paste detected flag

    Modifiers
      - left-shift usage count
      - right-shift usage count
      - capslock toggle count
      - capitals-via-shift count
      - capitals-via-capslock count

    Speed
      - typing speed in CPM
    """

    def __init__(self) -> None:
        self.pending_down: Dict[str, List[float]] = defaultdict(list)

        self.shift_left_active  = False
        self.shift_right_active = False
        self.capslock_on        = False

        self.dwell_by_code: Dict[str, List[float]] = defaultdict(list)

        self.global_dwell:     List[float] = []
        self.global_flight:    List[float] = []
        self.global_down_down: List[float] = []
        self.global_up_up:     List[float] = []

        self.printable_records: List[Dict[str, Any]] = []

        self.digraph_flight:    Dict[str, List[float]] = defaultdict(list)
        self.digraph_down_down: Dict[str, List[float]] = defaultdict(list)
        self.digraph_up_up:     Dict[str, List[float]] = defaultdict(list)
        self.trigraph_span:     Dict[str, List[float]] = defaultdict(list)

        self.backspace_count = 0
        self.delete_count    = 0
        self.paste_detected  = False

        self.left_shift_count      = 0
        self.right_shift_count     = 0
        self.capslock_toggle_count = 0
        self.capitals_via_shift    = 0
        self.capitals_via_capslock = 0

        self.last_keydown_t: Optional[float] = None
        self.last_keyup_t:   Optional[float] = None

        self.current_text = ""

    def process(self, events: List[Dict[str, Any]], target_text: str = "") -> Dict[str, Any]:
        if not events:
            return {"flat_features": {}, "feature_names": [], "feature_vector": []}

        events = sorted(events, key=lambda e: float(e.get("t", 0.0)))
        start_t     = float(events[0].get("t", 0.0))
        end_t       = float(events[-1].get("t", 0.0))
        duration_ms = max(1.0, end_t - start_t)

        # ── pass 1: raw event loop ────────────────────────────────────────
        for ev in events:
            etype = normalize_string(
                ev.get("type") or ev.get("eventType") or ev.get("eventtype")
            )
            t    = float(ev.get("t", 0.0))
            key  = ev.get("key")
            code = normalize_code(ev.get("code"), key)

            if etype == "paste":
                self.paste_detected = True
                continue

            if etype == "input":
                val = ev.get("value", "")
                self.current_text = str(val) if val is not None else self.current_text
                continue

            if etype == "keydown":
                norm_key = normalize_string(key)

                if norm_key == "backspace":
                    self.backspace_count += 1
                elif norm_key == "delete":
                    self.delete_count += 1

                if code == "shiftleft":
                    self.shift_left_active = True
                    self.left_shift_count += 1
                elif code == "shiftright":
                    self.shift_right_active = True
                    self.right_shift_count += 1
                elif code == "capslock":
                    self.capslock_on = not self.capslock_on
                    self.capslock_toggle_count += 1

                # BUG FIX: check raw key (not the lowercased sym) for isupper().
                # printable_symbol_from_event always returns lowercase, so
                # sym.isupper() was always False and capitals were never counted.
                raw_key = ev.get("key") or ev.get("Key") or ""
                if isinstance(raw_key, str) and len(raw_key) == 1 and raw_key.isupper() and raw_key != " ":
                    if self.shift_left_active or self.shift_right_active:
                        self.capitals_via_shift += 1
                    elif self.capslock_on:
                        self.capitals_via_capslock += 1

                if self.last_keydown_t is not None:
                    self.global_down_down.append(t - self.last_keydown_t)
                self.last_keydown_t = t

                is_repeat = ev.get("repeat") is True or ev.get("Repeat") is True
                sym = printable_symbol_from_event(ev)
                if sym and not is_repeat:
                    self.printable_records.append({
                        "symbol": sym,
                        "code":   code,
                        "down_t": t,
                        "up_t":   None,
                    })

                self.pending_down[code].append(t)

            elif etype == "keyup":
                if code == "shiftleft":
                    self.shift_left_active = False
                elif code == "shiftright":
                    self.shift_right_active = False

                if self.pending_down[code]:
                    dwell = t - self.pending_down[code].pop()
                    self.dwell_by_code[code].append(dwell)
                    self.global_dwell.append(dwell)

                if self.last_keyup_t is not None:
                    self.global_up_up.append(t - self.last_keyup_t)
                self.last_keyup_t = t

                for rec in reversed(self.printable_records):
                    if rec["code"] == code and rec["up_t"] is None:
                        rec["up_t"] = t
                        break

        # ── pass 2: n-gram intervals ──────────────────────────────────────
        seq = [r for r in self.printable_records if r["up_t"] is not None]

        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            k2   = make_ngram_key([a["symbol"], b["symbol"]])

            flight    = b["down_t"] - a["up_t"]
            down_down = b["down_t"] - a["down_t"]
            up_up     = b["up_t"]   - a["up_t"]

            self.digraph_flight[k2].append(flight)
            self.digraph_down_down[k2].append(down_down)
            self.digraph_up_up[k2].append(up_up)
            self.global_flight.append(flight)

        for i in range(len(seq) - 2):
            a, b, c = seq[i], seq[i + 1], seq[i + 2]
            k3 = make_ngram_key([a["symbol"], b["symbol"], c["symbol"]])
            self.trigraph_span[k3].append(c["down_t"] - a["up_t"])

        # ── assemble feature dict ─────────────────────────────────────────
        char_count       = len(self.current_text) if self.current_text else len(target_text)
        typing_speed_cpm = (char_count / duration_ms * 60_000.0) if duration_ms > 0 else 0.0

        attempt_features: Dict[str, Any] = {
            "meta": {
                "duration_ms":      duration_ms,
                "typing_speed_cpm": typing_speed_cpm,
            },
            "timings": {
                "dwell":     safe_mean_std(self.global_dwell),
                "flight":    safe_mean_std(self.global_flight),
                "down_down": safe_mean_std(self.global_down_down),
                "up_up":     safe_mean_std(self.global_up_up),
            },
            "errors": {
                "backspace_count": float(self.backspace_count),
                "delete_count":    float(self.delete_count),
                "paste_detected":  1.0 if self.paste_detected else 0.0,
            },
            "modifiers": {
                "left_shift_count":      float(self.left_shift_count),
                "right_shift_count":     float(self.right_shift_count),
                "capslock_toggle_count": float(self.capslock_toggle_count),
                "capitals_via_shift":    float(self.capitals_via_shift),
                "capitals_via_capslock": float(self.capitals_via_capslock),
            },
        }

        # BUG FIX: prefix per-key dwell as "dwell_key_" so the normalisation
        # pass (which checks for "dwell" in key name) correctly normalises them.
        # Previously the prefix was "key_" which was not caught by the TIME_KEYS
        # check, leaving raw millisecond values in the feature vector.
        for code, values in self.dwell_by_code.items():
            attempt_features[f"dwell_key_{code}"] = safe_mean_std(values)

        # per-digraph: flight + down-down + up-up + frequency
        for k2 in set(self.digraph_flight) | set(self.digraph_down_down) | set(self.digraph_up_up):
            fl  = self.digraph_flight.get(k2, [])
            dd  = self.digraph_down_down.get(k2, [])
            uu  = self.digraph_up_up.get(k2, [])
            attempt_features[f"digraph_{k2}"] = {
                "flight":    safe_mean_std(fl),
                "down_down": safe_mean_std(dd),
                "up_up":     safe_mean_std(uu),
                "frequency": float(max(len(fl), len(dd), len(uu))),
            }

        # per-trigraph: span + frequency
        for k3, spans in self.trigraph_span.items():
            attempt_features[f"trigraph_{k3}"] = {
                "span":      safe_mean_std(spans),
                "frequency": float(len(spans)),
            }

        # ── flatten & normalise by avg_dwell (per spec §1) ────────────────
        flat = flatten_numeric(attempt_features)

        avg_dwell = attempt_features["timings"]["dwell"]["mean"] or 1.0
        # All keys whose values are time-domain (milliseconds) get divided by
        # avg_dwell to make them session-speed-invariant.
        TIME_KEYS = ("duration_ms", "dwell", "flight", "down_down", "up_up", "span")

        normalized_flat = {
            k: (v / avg_dwell if any(tk in k for tk in TIME_KEYS) else v)
            for k, v in flat.items()
        }

        feature_names  = sorted(normalized_flat.keys())
        feature_vector = [normalized_flat[k] for k in feature_names]

        return {
            "flat_features":  normalized_flat,
            "feature_names":  feature_names,
            "feature_vector": feature_vector,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out_attempts = []
    phrase = payload.get("phrase", "")

    for idx, attempt in enumerate(payload.get("attempts", [])):
        extractor = AttemptExtractor()
        feats = extractor.process(attempt.get("events", []), target_text=phrase)
        out_attempts.append({
            "attemptId": attempt.get("attemptId", f"att_{idx}"),
            "features":  feats,
        })

    return {
        "userId":   payload.get("userId"),
        "phrase":   phrase,
        "attempts": out_attempts,
    }


def authenticate_user(parsed_json: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    model = KeystrokeModel.load(model_path)

    if not parsed_json.get("attempts"):
        return {"accepted": False, "error": "No attempts found"}

    features = parsed_json["attempts"][0]["features"]["flat_features"]
    result   = model.predict(features)

    return {
        "accepted":   result["accepted"],
        "score":      result["score"],
        "threshold":  result["threshold"],
        "confidence": result.get("confidence", 0.0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("--mode",  choices=["train", "verify"], default="train")
    ap.add_argument("--model", default="keystroke_model.pkl")
    args = ap.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.load(sys.stdin)

    processed_data = transform_payload(payload)

    if args.mode == "train":
        vectors, names = extract_training_data(processed_data)
        if not vectors:
            print("Error: No training data extracted")
            return
        model = KeystrokeModel()
        model.fit(vectors, names)
        model.save(args.model)
        print(f"Model trained on {len(vectors)} attempts and saved to {args.model}")
    else:
        result = authenticate_user(processed_data, args.model)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
