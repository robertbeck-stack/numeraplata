"""Plötuákvörðun, staðfesting milli ramma og skráningarstjórnun."""
from datetime import datetime
import os
import time
import numpy as np

from .plates import valid_plate, is_same_car
from .objects import detect_vehicles
from .storage import save_snapshot, save_logs, CAPTURES_DIR

MIN_READ_CONF = 0.45
CONFIRM_HITS = 3
CONFIRM_WINDOW = 6.0
CONFIRM_MIN_CONF = 0.65
RESIGHT_SECONDS = 60.0
VEHICLE_CHECK = True
VEHICLE_BYPASS_CONF = 0.85
REJECT_COOLDOWN = 10.0

_pending: dict[str, dict] = {}
_rejected: dict[str, float] = {}


def _confirm(plate: str, conf: float, frame, box, now: float, pending_dict=None):
    """Skráir lestur; skilar (besta öryggi, rammi, box) þegar platan er staðfest, annars None."""
    pending = _pending if pending_dict is None else pending_dict
    for p in [p for p, c in pending.items() if now - c["hits"][-1][0] > CONFIRM_WINDOW]:
        del pending[p]
    cand = pending.setdefault(plate, {"hits": [], "best": None})
    cand["hits"] = [t for t in cand["hits"] if now - t[0] <= CONFIRM_WINDOW] + [(now, conf)]
    if cand["best"] is None or conf > cand["best"][0]:
        cand["best"] = (conf, frame.copy(), box)
    confs = [c for _, c in cand["hits"]]
    if len(confs) >= CONFIRM_HITS and sum(confs) / len(confs) >= CONFIRM_MIN_CONF:
        del pending[plate]
        return cand["best"]
    return None


def plate_on_vehicle(frame, box) -> bool:
    if box is None:
        return False
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    for r in detect_vehicles(frame):
        v = r.bounding_box
        pad_x, pad_y = (v.x2 - v.x1) * 0.05, (v.y2 - v.y1) * 0.05
        if v.x1 - pad_x <= cx <= v.x2 + pad_x and v.y1 - pad_y <= cy <= v.y2 + pad_y:
            return True
    return False


def process_frame(alpr, frame, logs: list[dict], results=None, time_module=None,
                  pending_dict=None, vehicle_check: bool = None, resight_sec: float = None) -> list[dict]:
    """Vinnur úr ramma eða forreiknuðum niðurstöðum, staðfestir plötur og uppfærir logs."""
    t_mod = time if time_module is None else time_module
    now = t_mod.time()
    v_check = VEHICLE_CHECK if vehicle_check is None else vehicle_check
    r_sec = RESIGHT_SECONDS if resight_sec is None else resight_sec
    pending = _pending if pending_dict is None else pending_dict
    changed = False

    if results is None:
        results = alpr.predict(frame)

    reads: dict[str, tuple] = {}
    for res in results:
        if not (res.ocr and res.ocr.text):
            continue
        c = res.ocr.confidence
        avg_conf = float(np.mean(c)) if isinstance(c, (list, np.ndarray)) else float(c or 0)
        if avg_conf < MIN_READ_CONF:
            continue
        plate = valid_plate(res.ocr.text)
        if plate and (plate not in reads or avg_conf > reads[plate][0]):
            reads[plate] = (avg_conf, getattr(res.detection, "bounding_box", None))

    for plate, (avg_conf, read_box) in reads.items():
        confirmed = _confirm(plate, avg_conf, frame, read_box, now, pending_dict=pending)
        if confirmed is None:
            continue
        best_conf, frame_best, box = confirmed
        if v_check and best_conf < VEHICLE_BYPASS_CONF:
            if now - _rejected.get(plate, 0) < REJECT_COOLDOWN:
                continue
            if not plate_on_vehicle(frame_best, box):
                _rejected[plate] = now
                print(f"Hafnað (ekki á bíl, {best_conf:.0%}): {plate}")
                continue

        conf = round(best_conf * 100)
        match_idx = next(
            (i for i, e in enumerate(logs)
             if now - e["ts_sec"] < r_sec and (e["plate"] == plate or is_same_car(e["plate"], plate))),
            -1,
        )
        if match_idx != -1:
            logs[match_idx]["ts_sec"] = now
        if match_idx == -1:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            img_url = save_snapshot(frame_best, box, plate, ts)
            logs.insert(0, {"time": ts, "plate": plate, "conf": conf, "img": img_url, "ts_sec": now})
            changed = True
        elif conf > logs[match_idx]["conf"] and logs[match_idx]["plate"] != plate:
            old = logs[match_idx]
            stale = old["img"].lstrip("/\\")
            old["plate"] = plate
            old["conf"] = conf
            old["img"] = save_snapshot(frame_best, box, plate, old["time"])
            if stale != old["img"].lstrip("/\\") and stale.startswith(f"{CAPTURES_DIR}/") and not any(e["img"].lstrip("/\\") == stale for e in logs):
                try:
                    os.remove(stale)
                except OSError:
                    pass
            old["ts_sec"] = now
            changed = True

    if changed:
        save_logs(logs)
    return logs if changed else []
