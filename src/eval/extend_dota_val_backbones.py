"""
Extend the DOTA-val counting result to all 3 paper backbones.
Reuses the stored detector counts + GT in
dota_val_counting.json; adds Qwen3-VL and InternVL3.5 counts per cell.
Qwen3-VL is a thinking model -> chat API (answer in message.content after thinking);
InternVL3.5 -> plain generate. Checkpointed.
"""
import base64
import io
import json
import re
import sys
from pathlib import Path

import requests

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = Path("E:/FP-Tag/DOTAv1/images/val")
DATA = ROOT / "experiments/dota_val_counting.json"
NOUN = {"plane": "airplanes", "ship": "ships",
        "small vehicle": "small vehicles (cars)",
        "large vehicle": "large vehicles (trucks or buses)"}


def b64(stem):
    return base64.b64encode(open(IMG_DIR / f"{stem}.jpg", "rb").read()).decode()


def count_generate(model, img, cname):
    prompt = f"How many {NOUN[cname]} are in this image? Answer with a single integer only."
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": model, "prompt": prompt, "images": [img], "stream": False,
            "options": {"temperature": 0, "num_predict": 48, "num_gpu": 99}}, timeout=300)
        m = re.search(r"\b(\d+)\b", r.json().get("response", ""))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def count_chat_thinking(model, img, cname):
    prompt = f"How many {NOUN[cname]} are in this image? Answer with a single integer only."
    try:
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": model, "messages": [{"role": "user", "content": prompt, "images": [img]}],
            "stream": False, "options": {"temperature": 0, "num_predict": 1024, "num_gpu": 99}},
            timeout=300)
        msg = r.json().get("message", {})
        txt = msg.get("content", "") or str(msg.get("thinking", ""))
        m = re.search(r"\b(\d+)\b", txt)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    rows = d["rows"]
    images = sorted(set(r["image"] for r in rows))
    # group rows by image for batched image encode
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)

    # Process ONE backbone fully before the other to avoid Ollama model swaps
    # (alternating models reloads 8B weights every call -> catastrophically slow).
    def run_backbone(field, fn, model):
        for k, stem in enumerate(images):
            cells = by_img[stem]
            if all(field in c for c in cells):
                continue
            img = b64(stem)
            for c in cells:
                if field not in c:
                    c[field] = fn(model, img, c["cls"])
            if k % 10 == 0:
                json.dump(d, open(DATA, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                print(f"  [{field}] {k+1}/{len(images)} images", flush=True)
        json.dump(d, open(DATA, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("=== Pass 1: InternVL3.5 (generate) ===", flush=True)
    run_backbone("vlm_ivl", count_generate, "blaifa/InternVL3_5:8b")
    print("=== Pass 2: Qwen3-VL (chat/thinking) ===", flush=True)
    run_backbone("vlm_q3", count_chat_thinking, "qwen3-vl:8b")
    print(f"DONE all 3 backbones -> {DATA.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
