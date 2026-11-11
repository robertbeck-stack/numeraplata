"""Gagnageymsla fyrir númeraskráningar, myndir, stillingar og öruggar skráaraðgerðir."""
from datetime import datetime
import json
import os
import tempfile
import cv2
import numpy as np

LOG_FILE = "plates.csv"
TRAFFIC_FILE = "traffic.csv"
PEOPLE_FILE = "people.csv"
CAPTURES_DIR = "captures"
SETTINGS_FILE = "settings.json"


def is_safe_path(base_dir: str, requested_path: str) -> bool:
    """Koma í veg fyrir Path Traversal (árásir með ../ eða \\..).

    Athugar hvort markskráin sé raunverulega innan base_dir möppunnar.
    """
    try:
        base = os.path.abspath(base_dir)
        target = os.path.abspath(os.path.join(base_dir, requested_path.lstrip("/\\")))
        return os.path.commonpath([base]) == os.path.commonpath([base, target])
    except Exception:
        return False


def load_logs(path: str = LOG_FILE) -> list[dict]:
    """Les fyrri skráningar úr plates.csv (nýjasta fyrst) svo þær haldist milli keyrslna."""
    if not os.path.isfile(path):
        return []
    logs = []
    with open(path, encoding="utf-8") as f:
        next(f, None)  # haus
        for line in f:
            parts = line.rstrip("\r\n").split(",")
            if len(parts) < 5:
                continue
            day, clock, plate, conf, img = parts[:5]
            ts = f"{day} {clock}"
            try:
                ts_sec = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                continue
            logs.append({
                "time": ts,
                "plate": plate,
                "conf": int(conf.rstrip("%") or 0),
                "img": img,
                "ts_sec": ts_sec
            })
    logs.reverse()
    return logs


def save_logs(logs: list[dict], path: str = LOG_FILE) -> None:
    """Vistar skráningar í plates.csv með atómískum skrifum svo skráin skemmist aldrei."""
    dirname = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dirname, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=dirname, prefix="plates_", suffix=".tmp")
    try:
        with open(temp_fd, "w", encoding="utf-8") as f:
            f.write("Dagur,Tími,Bílnúmer,Öryggi,Mynd\n")
            for e in reversed(logs):
                day, clock = e["time"].split(" ")
                f.write(f"{day},{clock},{e['plate']},{e['conf']}%,{e['img']}\n")
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def delete_entry(logs: list[dict], time_str: str, plate: str) -> bool:
    """Eyðir einni færslu (og skyndimynd hennar) og uppfærir plates.csv."""
    idx = next((i for i, e in enumerate(logs) if e["time"] == time_str and e["plate"] == plate), -1)
    if idx == -1:
        return False
    entry = logs.pop(idx)
    img = entry["img"].lstrip("/\\")
    if is_safe_path(CAPTURES_DIR, img) and not any(e["img"].lstrip("/\\") == img for e in logs):
        try:
            os.remove(img)
        except OSError:
            pass
    save_logs(logs)
    return True


def save_snapshot(frame, box, plate: str, ts: str, captures_dir: str = CAPTURES_DIR) -> str:
    """Vistar útklippta skyndimynd af bíl og plötu í captures möppuna."""
    os.makedirs(captures_dir, exist_ok=True)
    annotated = frame.copy()
    if box is not None:
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        pad_x = max(6, int((x2 - x1) * 0.08))
        pad_y = max(6, int((y2 - y1) * 0.18))
        h, w = annotated.shape[:2]
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated, plate, (x1, max(32, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    fname = f"numby_{plate}_{ts.replace(':', '').replace(' ', '_')}.jpg"
    rel_path = f"{captures_dir}/{fname}"
    cv2.imwrite(rel_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return "/" + rel_path.replace("\\", "/")


def prune_captures(captures_dir: str = CAPTURES_DIR, max_days: int = 30, max_files: int = 5000) -> int:
    """Eyðir gömlum myndum í captures möppunni svo diskurinn fyllist ekki."""
    if not os.path.isdir(captures_dir):
        return 0
    import time
    now = time.time()
    cutoff = now - (max_days * 86400)
    files = []
    for fname in os.listdir(captures_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fpath = os.path.join(captures_dir, fname)
            try:
                stat = os.stat(fpath)
                files.append((stat.st_mtime, fpath))
            except OSError:
                pass
    files.sort()
    deleted = 0
    for mtime, fpath in files:
        if mtime < cutoff or (len(files) - deleted > max_files):
            try:
                os.remove(fpath)
                deleted += 1
            except OSError:
                pass
    return deleted


def clamp_zoom(z: dict) -> dict:
    zoom = min(4.0, max(1.0, float(z.get("zoom", 1))))
    half = 0.5 / zoom
    cx = min(1 - half, max(half, float(z.get("cx", 0.5))))
    cy = min(1 - half, max(half, float(z.get("cy", 0.5))))
    return {"zoom": round(zoom, 2), "cx": round(cx, 4), "cy": round(cy, 4)}


def crop_zoom(frame, z: dict):
    """Stafrænn aðdráttur: klippir út svæðið sem notandinn valdi. Greiningin keyrir á útklippunni,
    svo plötur verða stærri í augum líkansins (það minnkar allt niður í 384px)."""
    if z["zoom"] <= 1.0:
        return frame
    h, w = frame.shape[:2]
    cw, ch = int(w / z["zoom"]), int(h / z["zoom"])
    x0 = int(min(max(z["cx"] * w - cw / 2, 0), w - cw))
    y0 = int(min(max(z["cy"] * h - ch / 2, 0), h - ch))
    return np.ascontiguousarray(frame[y0:y0 + ch, x0:x0 + cw])
