"""
Feature engineering for fixed-text keystroke dynamics authentication.

Standard approach (Killourhy & Maxion 2009):
  For a phrase of length L the feature vector has 3L-2 dimensions:
    ht_i   – hold time at position i          (keydown_i  → keyup_i)
    dd_i   – down-down from position i to i+1 (keydown_i+1 - keydown_i)
    ud_i   – up-down (flight) from pos i→i+1  (keydown_i+1 - keyup_i)

For "The quick brown fox jumps over the lazy dog" (L=43): 127 features.

All timing values are divided by the mean hold time of the attempt so that
typing speed differences between sessions do not affect the score
(speed-invariant normalization, per spec §1).

Attempts where the reconstructed typed text does not match the phrase are
marked invalid and excluded from training / rejected on verification.
"""

import json
import sys
import argparse
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from app.ml_model.model import KeystrokeModel, extract_training_data


# ---------------------------------------------------------------------------
# Event-stream helpers
# ---------------------------------------------------------------------------

def _etype(ev: Dict[str, Any]) -> str:
    raw = (ev.get("type") or ev.get("eventType") or ev.get("eventtype") or "").lower().strip()
    # README2 shows both "keydown"/"keyup" and the short forms "down"/"up".
    # Normalise so the rest of the code only deals with the full names.
    if raw == "down":
        return "keydown"
    if raw == "up":
        return "keyup"
    return raw


def _code(ev: Dict[str, Any]) -> str:
    c = ev.get("code") or ev.get("Code") or ""
    k = ev.get("key")  or ev.get("Key")  or ""
    s = str(c).lower().strip()
    if s:
        return s
    if k == " ":
        return "space"
    return str(k).lower().strip() or "unknown"


def _is_printable(key: Any) -> bool:
    """True when key produces a single printable character (including space)."""
    return isinstance(key, str) and len(key) == 1 and key.isprintable()


# ---------------------------------------------------------------------------
# Text-buffer reconstruction
# ---------------------------------------------------------------------------

def extract_char_sequence(events: List[Dict[str, Any]]) -> Tuple[List[Dict], str]:
    """
    Simulate the browser text-input buffer from the raw event stream.

    Returns
    -------
    (records, typed_text) where:
      records[i] = {"char": str, "code": str, "down_t": float, "up_t": float}
                   for the i-th character in the FINAL typed text.
      typed_text  = "".join(r["char"] for r in records)

    Algorithm
    ---------
    1. Replay keydown/keyup events as a stack — printable keydown pushes,
       Backspace keydown pops, keyup fills in up_t for the matching entry.

    2. `input` events carry the authoritative DOM `value` after each change.
       We compare the FULL buffer text (including entries still waiting for
       their keyup) against the last seen `value`.  If they disagree the
       stream has drifted (autocorrect, IME, Delete key, etc.) and the
       attempt is rejected via a sentinel.

    3. Missing up_t (most commonly the very last character — the keyup fires
       after form submission and is never recorded): estimated from the median
       hold time of all other complete characters in the same attempt.

    Paste:    any paste event marks the buffer dirty (sentinel pushed).
    """
    events = sorted(events, key=lambda e: float(e.get("t", 0.0)))
    buf: List[Dict[str, Any]] = []
    last_input_value: Optional[str] = None

    for ev in events:
        et  = _etype(ev)
        key = ev.get("key") or ev.get("Key")
        cod = _code(ev)
        t   = float(ev.get("t", 0.0))

        if et == "paste":
            buf.append({"char": "\x00", "code": "paste", "down_t": t, "up_t": t})
            continue

        if et == "input":
            val = ev.get("value")
            if val is not None:
                last_input_value = str(val)
            continue

        if et == "keydown":
            if ev.get("repeat") or ev.get("Repeat"):
                continue

            norm = (str(key) if key else "").lower()
            if norm == "backspace":
                if buf:
                    buf.pop()
            elif norm == "delete":
                pass  # forward-delete: ignore (rare in fixed-text auth)
            elif _is_printable(key):
                buf.append({"char": key, "code": cod, "down_t": t, "up_t": None})

        elif et == "keyup":
            for entry in reversed(buf):
                if entry["code"] == cod and entry["up_t"] is None:
                    entry["up_t"] = t
                    break

    # Full reconstructed text including entries whose keyup hasn't arrived yet
    printable_buf = [e for e in buf if e["char"] != "\x00"]
    all_text = "".join(e["char"] for e in printable_buf)

    # Cross-check against DOM state.  The last input event value is authoritative.
    # A mismatch means the buffer reconstruction diverged (autocorrect, IME, etc.).
    if last_input_value is not None and all_text != last_input_value:
        return [], "\x00"

    # Fill missing up_t (typically only the final character, whose keyup fires
    # after the form submits).  Use median hold time of complete records as the
    # best available estimate.
    complete_holds = [e["up_t"] - e["down_t"] for e in printable_buf if e["up_t"] is not None]
    fallback_hold  = median(complete_holds) if complete_holds else 80.0
    for entry in printable_buf:
        if entry["up_t"] is None:
            entry["up_t"] = entry["down_t"] + fallback_hold

    return printable_buf, all_text


# ---------------------------------------------------------------------------
# Per-position feature extraction
# ---------------------------------------------------------------------------

def extract_phrase_features(
    events: List[Dict[str, Any]],
    phrase: str,
    final_text: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Extract per-position timing features for one attempt.

    Parameters
    ----------
    events     : raw event list from the attempt
    phrase     : the expected target phrase
    final_text : the `finalText` field sent by the frontend (README2 format).
                 When provided it is used as the primary validity check,
                 making the function faster and more reliable than pure
                 event-stream reconstruction.

    Returns a flat dict of normalised floats, or None when the attempt is
    invalid (typed text ≠ phrase, missing timestamps, non-positive hold times,
    or event-stream/DOM-state mismatch detected).

    Feature names
    -------------
    ht_{i:03d}   – normalised hold time at position i
    dd_{i:03d}   – normalised down-down interval, positions i → i+1
    ud_{i:03d}   – normalised up-down (flight) interval, positions i → i+1
    """
    if not events or not phrase:
        return None

    # Quick pre-filter using the frontend-computed finalText when available.
    # This avoids reconstructing the buffer for clearly wrong attempts.
    if final_text is not None and final_text != phrase:
        return None

    records, typed_text = extract_char_sequence(events)

    # Reject if sentinel (event-stream/DOM mismatch) or text doesn't match
    if typed_text == "\x00" or typed_text != phrase:
        return None

    L = len(phrase)
    if len(records) != L:
        return None

    # Raw timings
    ht = [records[i]["up_t"] - records[i]["down_t"] for i in range(L)]
    dd = [records[i + 1]["down_t"] - records[i]["down_t"] for i in range(L - 1)]
    ud = [records[i + 1]["down_t"] - records[i]["up_t"]  for i in range(L - 1)]

    # All hold times must be positive (sanity check for clock glitches)
    if any(h <= 0 for h in ht):
        return None

    # Speed-invariant normalisation: divide all intervals by mean hold time
    avg_ht = sum(ht) / L
    if avg_ht <= 0:
        return None

    features: Dict[str, float] = {}
    for i, v in enumerate(ht):
        features[f"ht_{i:03d}"] = v / avg_ht
    for i, v in enumerate(dd):
        features[f"dd_{i:03d}"] = v / avg_ht
    for i, v in enumerate(ud):
        features[f"ud_{i:03d}"] = v / avg_ht

    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process all attempts in a payload dict.

    Each output attempt includes a "features" sub-dict with:
      flat_features  – {feature_name: float} or {} when invalid
      feature_names  – sorted list of names
      feature_vector – values in the same order
      valid          – bool: True only when typed text matched the phrase
    """
    phrase = payload.get("phrase", "")
    out_attempts = []

    for idx, attempt in enumerate(payload.get("attempts", [])):
        # Pass finalText from the attempt when present (README2 format).
        # extract_phrase_features uses it as a fast pre-filter before
        # running the more expensive buffer-reconstruction pass.
        final_text = attempt.get("finalText") or attempt.get("final_text")
        features = extract_phrase_features(
            attempt.get("events", []), phrase, final_text=final_text
        )
        valid = features is not None

        out_attempts.append({
            "attemptId": attempt.get("attemptId", f"att_{idx}"),
            "features": {
                "flat_features":  features if valid else {},
                "feature_names":  sorted(features.keys()) if valid else [],
                "feature_vector": [features[k] for k in sorted(features.keys())] if valid else [],
                "valid":          valid,
            },
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

    attempt = parsed_json["attempts"][0]
    if not attempt["features"].get("valid"):
        return {"accepted": False, "error": "Typed text did not match the phrase"}

    result = model.predict(attempt["features"]["flat_features"])
    return result


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
            print("Error: No valid training attempts (check that typed text matches phrase)")
            return
        model = KeystrokeModel()
        model.fit(vectors, names)
        model.save(args.model)
        print(f"Model trained on {len(vectors)} attempts → {args.model}")
    else:
        result = authenticate_user(processed_data, args.model)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
