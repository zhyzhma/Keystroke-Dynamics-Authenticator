import json
import sys
import argparse
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional

# Импортируем обновленный класс и вспомогательные функции
from app.ml_model.model import KeystrokeModel, extract_training_data

NG_SEP = "␟"

def authenticate_user(parsed_json: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """
    Проводит аутентификацию, используя словарь признаков.
    """
    model = KeystrokeModel()
    model.load(model_path)

    if not parsed_json.get("attempts"):
        return {"accepted": False, "error": "No attempts found"}
    
    # Берем признаки из первой попытки для верификации
    first_attempt_features = parsed_json["attempts"][0]["features"]["flat_features"]
    
    # Модель сама сопоставит ключи (уже приведенные к нижнему регистру)
    result = model.predict(first_attempt_features)

    return {
        "accepted": result["accepted"],
        "score": result["score"],
        "threshold": result["threshold"],
        "confidence": result.get("confidence", 0.0)
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

def normalize_string(s: Any) -> str:
    """Приводит строку к нижнему регистру и удаляет лишние пробелы."""
    if s is None:
        return ""
    return str(s).strip().lower()

def normalize_code(code: Optional[str], key: Optional[str]) -> str:
    """Нормализует код клавиши к нижнему регистру."""
    if code:
        return normalize_string(code)
    
    if key == " ":
        return "space"
    
    if key:
        return normalize_string(key)
    
    return "unknown"

def printable_symbol_from_event(ev: Dict[str, Any]) -> Optional[str]:
    """Извлекает символ, приводя его к нижнему регистру для единообразия."""
    # Обрабатываем возможные варианты написания ключей в JSON
    key = ev.get("key") or ev.get("Key")
    code = ev.get("code") or ev.get("Code")
    
    #norm_key = normalize_string(key)
    norm_code = normalize_code(code, key)
    
    if norm_code == "space" or key == " ":
        return " "
    
    if isinstance(key, str) and len(key) == 1:
        return key.lower()
    
    return None

def make_ngram_key(tokens: List[str]) -> str:
    return NG_SEP.join(tokens)

def flatten_numeric(d: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            # Ключи признаков тоже приводим к нижнему регистру
            clean_k = k.lower()
            key = f"{prefix}{clean_k}" if not prefix else f"{prefix}.{clean_k}"
            
            if isinstance(v, dict):
                out.update(flatten_numeric(v, key))
            elif isinstance(v, (int, float, bool)) and not isinstance(v, str):
                out[key] = float(v)
    return out

class AttemptExtractor:
    def __init__(self) -> None:
        self.pending_down = defaultdict(list)
        # Названия модификаторов в нижнем регистре
        self.active_mods = {"shiftleft": False, "shiftright": False, "capslock": False}
        self.dwell_by_code = defaultdict(list)
        self.global_dwell = []
        self.printable_records = []
        self.digraphs = defaultdict(lambda: defaultdict(list))
        self.trigraphs = defaultdict(lambda: defaultdict(list))
        self.total_keydowns = 0
        self.backspace_count = 0
        self.current_text = ""

    def process(self, events: List[Dict[str, Any]], target_text: str = "") -> Dict[str, Any]:
        if not events:
            return {"flat_features": {}, "feature_names": [], "feature_vector": []}

        events = sorted(events, key=lambda e: float(e.get("t", 0.0)))
        start_t = float(events[0].get("t", 0.0))
        end_t = float(events[-1].get("t", 0.0))
        duration_ms = max(1.0, end_t - start_t)

        for ev in events:
            # Гибкий поиск типа события
            etype = normalize_string(ev.get("type") or ev.get("eventType") or ev.get("eventtype"))
            t = float(ev.get("t", 0.0))
            key = ev.get("key")
            code = normalize_code(ev.get("code"), key)
            
            if etype == "keydown":
                self.total_keydowns += 1
                if normalize_string(key) == "backspace":
                    self.backspace_count += 1
                
                if code in self.active_mods:
                    if code == "capslock":
                        self.active_mods[code] = not self.active_mods[code]
                    else:
                        self.active_mods[code] = True
                
                sym = printable_symbol_from_event(ev)
                # Игнорируем повторы при зажатии (key repeat) для чистоты таймингов
                is_repeat = ev.get("repeat") is True or ev.get("Repeat") is True
                if sym and not is_repeat:
                    self.printable_records.append({"symbol": sym, "code": code, "down_t": t, "up_t": None})
                
                self.pending_down[code].append(t)

            elif etype == "keyup":
                if code == "shiftleft" or code == "shiftright":
                    self.active_mods[code] = False
                
                if self.pending_down[code]:
                    dwell = t - self.pending_down[code].pop()
                    self.dwell_by_code[code].append(dwell)
                    self.global_dwell.append(dwell)
                
                for rec in reversed(self.printable_records):
                    if rec["code"] == code and rec["up_t"] is None:
                        rec["up_t"] = t
                        break
            
            elif etype == "input":
                val = ev.get("value", "")
                self.current_text = str(val)

        # Расчет интервалов между клавишами (диграфы)
        seq = [r for r in self.printable_records if r["up_t"] is not None]
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            k2 = make_ngram_key([a["symbol"], b["symbol"]])
            self.digraphs[k2]["flight"].append(b["down_t"] - a["up_t"])
        
        # Расчет триграфов (интервалы для 3-х клавиш)
        for i in range(len(seq) - 2):
            a, b, c = seq[i], seq[i + 1], seq[i + 2]
            k3 = make_ngram_key([a["symbol"], b["symbol"], c["symbol"]])
            # Интервал от отпускания первой до нажатия третьей (один из вариантов признака)
            self.trigraphs[k3]["total_time"].append(c["down_t"] - a["up_t"])
        
        
        char_count = len(self.current_text) if self.current_text else len(target_text)
        typing_speed_cpm = (char_count / duration_ms * 60000.0) if duration_ms > 0 else 0.0

        attempt_features = {
            "meta": {
                "duration_ms": duration_ms, 
                "typing_speed_cpm": typing_speed_cpm
            },
            "timings": {
                "dwell": safe_mean_std(self.global_dwell),
                "flight": safe_mean_std([d for g in self.digraphs.values() for d in g["flight"]])
            },
            "errors": {
                "backspace_count": self.backspace_count
            }
        }

        # Добавляем dwell-time для каждой физической клавиши (в нижнем регистре)
        for code, values in self.dwell_by_code.items():
            attempt_features[f"key_{code}"] = safe_mean_std(values)

        for code, values in self.trigraphs.items():
            attempt_features[f"trigraph_{code}"] = safe_mean_std(values["flight_total"])


        flat = flatten_numeric(attempt_features)
        
        # Нормализация относительно средней скорости нажатия
        avg_dwell = attempt_features["timings"]["dwell"]["mean"]
        if not avg_dwell:
            avg_dwell = 1.0
            
        normalized_flat = {}
        for k, v in flat.items():
            # Если в названии признака есть намек на время, нормируем его
            if "ms" in k or "dwell" in k or "flight" in k:
                normalized_flat[k] = v / avg_dwell
            else:
                normalized_flat[k] = v

        return {
            "flat_features": normalized_flat,
            "feature_names": sorted(normalized_flat.keys()),
            "feature_vector": [normalized_flat[k] for k in sorted(normalized_flat.keys())]
        }

def transform_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    attempts = payload.get("attempts", [])
    out_attempts = []
    
    for idx, attempt in enumerate(attempts):
        extractor = AttemptExtractor()
        events = attempt.get("events", [])
        phrase = payload.get("phrase", "")
        
        feats = extractor.process(events, target_text=phrase)
        
        out_attempts.append({
            "attemptId": attempt.get("attemptId", f"att_{idx}"),
            "features": feats
        })
        
    return {
        "userId": payload.get("userId"),
        "phrase": payload.get("phrase"),
        "attempts": out_attempts
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("--mode", choices=["train", "verify"], default="train")
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