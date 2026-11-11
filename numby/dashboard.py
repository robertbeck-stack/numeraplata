"""Meginstjórnborð (Dashboard) fyrir Numby ALPR: Lifandi straumur, vefviðmót og OpenCV gluggi."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import webbrowser
from types import SimpleNamespace as NS
import cv2
import numpy as np

from .scanner import make_alpr, MIN_SCAN_PERIOD, IDLE_SCAN_PERIOD, IDLE_VEHICLE_INTERVAL, _raise_thread_priority
from .objects import detect_objects, VEHICLE_LABELS, VEHICLE_INTERVAL
from .tracking import BoxTracker
from .traffic import VehicleTracks, NightCounter, COUNT_LINE, MOVING_VEHICLE_INTERVAL
from .motion import MotionTracker
from .people import PeopleCounter
from .storage import (
    load_logs, delete_entry, clamp_zoom, crop_zoom, is_safe_path,
    LOG_FILE, TRAFFIC_FILE, PEOPLE_FILE, CAPTURES_DIR, SETTINGS_FILE
)
from .pipeline import process_frame
from .drawing import (
    draw_vehicles, draw_hidden, draw_people, draw_results, build_panel,
    clean_text_for_opencv
)
from .server import DASHBOARD_HTML, _send_file
from . import VERSION

WINDOW = f"Numby ALPR ({VERSION}) - q til ad haetta"
DISPLAY_MAX_H = 540


def dashboard(source: str | int = 0, host: str = "127.0.0.1", port: int = 8000, window: bool = False) -> None:
    """Les myndavél/straum í Python og sýnir lifandi mynd + skráð númer í vafra (eða í glugga með window=True)."""
    alpr, logs = make_alpr(), load_logs()
    lock = threading.Lock()
    state = {"jpg": None, "live": False, "res": "", "fps": 0.0, "seq": 0, "snap": None}
    snap_lock = threading.Lock()

    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            zoom = clamp_zoom(json.load(f))
    except (OSError, ValueError):
        zoom = clamp_zoom({})
    state["zoom"] = zoom

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Gat ekki opnað myndavél/straum: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    state["res"] = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"

    is_file = isinstance(source, str) and not source.isdigit() and "://" not in source
    frame_period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0.0

    def capture_loop():
        """Les myndavélina og endurtengist sjálfkrafa við netrof."""
        nonlocal cap
        _raise_thread_priority()
        frame_idx = 0
        fail_count = 0
        while True:
            t_read = time.time()
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                state["live"] = False
                time.sleep(0.5)
                if fail_count >= 10 and not is_file:  # ~5 sek án ramma: reyna endurtengingu
                    try:
                        cap.release()
                        time.sleep(1.0)
                        new_cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
                        if new_cap.isOpened():
                            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                            cap = new_cap
                            fail_count = 0
                    except Exception:
                        pass
                continue
            fail_count = 0
            frame_idx += 1
            z = state["zoom"]
            frame = crop_zoom(frame, z)
            state["frame"], state["live"] = (frame_idx, frame, z), True
            traffic.night = night.update(frame, time.time())
            # hreyfirakning á hverjum ramma (að degi til): fylgir bílum á ferð án þess að bíða eftir bílagreiningunni
            movers = [] if traffic.night else motion.update(frame, time.time(), frame_idx)
            det = state["det"]
            if det is not None and det[0] != seen_det[0]:  # ný bílagreining: para hana við hreyfihlutina
                seen_det[0] = det[0]
                motion.assign(det[2], det[0])
            req = state["snap"]
            snap = req is not None and not req["done"].is_set()
            if snap and req["raw"]:
                req["frame"] = frame
                req["done"].set()
                snap = False
            now = time.time()
            watching = snap or window or (now - state["watch"] < 2.0)
            if watching:
                objs = tracker.update(frame_idx, frame, state["det"], z)
                view = frame.copy()
                draw_people(view, [o for o in objs if o.label == "person"])
                vehicles = [o for o in objs if o.label in VEHICLE_LABELS and o.tid]
                still, moving = motion.merge(vehicles, movers)
                draw_vehicles(view, still)
                draw_vehicles(view, moving)
                lx = int(COUNT_LINE * view.shape[1])
                for y in range(0, view.shape[0], 24):
                    cv2.line(view, (lx, y), (lx, y + 10), (255, 255, 255), 1)
                shown = {getattr(o, "tid", None) for o in objs}
                draw_hidden(view, [h for h in traffic.hidden(time.time()) if h.tid not in shown])
                draw_results(view, [
                    NS(detection=o, ocr=NS(text=o.label, confidence=o.confidence) if o.label else None)
                    for o in plate_tracker.update(frame_idx, frame, state["plates"], z, anchors=objs)
                ])
                # Útgáfunúmer efst vinstra megin
                (tw, th), _ = cv2.getTextSize(VERSION, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(view, (10, 10), (10 + tw + 14, 10 + th + 12), (20, 20, 20), -1)
                cv2.rectangle(view, (10, 10), (10 + tw + 14, 10 + th + 12), (60, 60, 60), 1)
                cv2.putText(view, VERSION, (17, 10 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)
                if snap:
                    req["frame"] = view.copy()
                    req["done"].set()
                h, w = view.shape[:2]
                if h > 720:
                    view = cv2.resize(view, (int(w * 720 / h), 720), interpolation=cv2.INTER_AREA)
                state["view"] = view
                ok, jpg = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    state["jpg"], state["seq"] = jpg.tobytes(), state["seq"] + 1
            if frame_period:
                time.sleep(max(0.0, frame_period - (time.time() - t_read)))

    def plate_loop():
        last, t0, n = None, time.time(), 0
        while True:
            item = state["frame"]
            if item is None or item is last:
                time.sleep(0.004)
                continue
            last, t_scan = item, time.time()
            frame_idx, frame, z = item
            try:
                results = alpr.predict(frame)
                state["plates"] = (frame_idx, z, [
                    NS(label=r.ocr.text if r.ocr else "", bounding_box=r.detection.bounding_box,
                       confidence=float(np.mean(r.ocr.confidence)) if r.ocr else 0.0) for r in results
                ])
                with lock:
                    if process_frame(alpr, frame, logs, results):
                        top = logs[0]
                        print(f"{top['time']}\t{top['plate']}\t({top['conf']}%) -> {top['img']}")
            except Exception as e:
                print(f"Plötugreining mistókst: {e}")
                time.sleep(1)
            n += 1
            if time.time() - t0 >= 2:
                state["fps"], t0, n = round(n / (time.time() - t0), 1), time.time(), 0
            time.sleep(max(0.0, (MIN_SCAN_PERIOD if alpr.busy else IDLE_SCAN_PERIOD) - (time.time() - t_scan)))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        state["watch"] = time.time()
                        if state["seq"] == seq or state["jpg"] is None:
                            time.sleep(0.02)
                            continue
                        seq, jpg = state["seq"], state["jpg"]
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (ConnectionError, BrokenPipeError, OSError):
                    return
            if path == "/zoom":
                self._json(state["zoom"])
                return
            if path == "/events":
                with lock:
                    events = [{k: v for k, v in e.items() if k != "ts_sec"} for e in logs]
                body = json.dumps({
                    "events": events,
                    "live": state["live"],
                    "res": state["res"],
                    "fps": state["fps"],
                    "traffic": {"left": traffic.left, "right": traffic.right},
                    "people": {"now": counter.now, "peak": counter.peak, "peak_time": counter.peak_time}
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/plates.csv" and _send_file(self, LOG_FILE, "text/csv; charset=utf-8"):
                return
            if path == "/traffic.csv":
                if not _send_file(self, TRAFFIC_FILE, "text/csv; charset=utf-8"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.end_headers()
                    self.wfile.write("Dagur,Tími,Tegund,Stefna\n".encode("utf-8"))
                return
            if path == "/people.csv":
                if not _send_file(self, PEOPLE_FILE, "text/csv; charset=utf-8"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.end_headers()
                    self.wfile.write("Dagur,Tími,Fjöldi\n".encode("utf-8"))
                return
            clean_path = path.lstrip("/\\")
            if clean_path.startswith(CAPTURES_DIR) and is_safe_path(CAPTURES_DIR, clean_path):
                if _send_file(self, clean_path, "image/jpeg"):
                    return
            if path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
                return
            self.send_error(404)

        def do_POST(self):
            if self.path == "/delete":
                try:
                    d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    with lock:
                        ok = delete_entry(logs, str(d["time"]), str(d["plate"]))
                except (ValueError, KeyError, TypeError):
                    self.send_error(400)
                    return
                if ok:
                    self._json({"ok": True})
                else:
                    self.send_error(404)
                return
            if self.path.split("?")[0] == "/photo":
                with snap_lock:
                    req = {"done": threading.Event(), "frame": None, "raw": "raw=1" in self.path}
                    state["snap"] = req
                    got = req["done"].wait(3.0)
                    state["snap"] = None
                if not got:
                    self.send_error(503, "Engin mynd frá myndavél")
                    return
                os.makedirs(CAPTURES_DIR, exist_ok=True)
                name = f"mynd_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.jpg"
                if not req["raw"]:
                    z = state["zoom"]
                    note = clean_text_for_opencv(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  aðdráttur {z['zoom']:.1f}x  |  "
                        f"{state['fps']} skannanir/s  |  {'nótt (ljósatalning)' if traffic.night else 'dagur'}  |  "
                        f"bílar í dag < {traffic.left}  {traffic.right} >  |  fólk {counter.now}"
                    )
                    fh = req["frame"].shape[0]
                    (tw, th), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(req["frame"], (0, fh - th - 24), (tw + 24, fh), (20, 20, 20), -1)
                    cv2.putText(req["frame"], note, (12, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(CAPTURES_DIR, name), req["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                h, w = req["frame"].shape[:2]
                print(f"Mynd vistuð: {CAPTURES_DIR}/{name} ({w}×{h})")
                self._json({"img": f"/{CAPTURES_DIR}/{name}", "name": name, "res": f"{w}×{h}"})
                return
            if self.path != "/zoom":
                self.send_error(404)
                return
            try:
                z = clamp_zoom(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"))
            except (ValueError, TypeError):
                self.send_error(400)
                return
            state["zoom"] = z
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(z, f)
            self._json(z)

        def _json(self, obj):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(obj).encode())

        def log_message(self, *args):
            pass

    def vehicle_loop():
        last, last_run = None, 0.0
        while True:
            item = state["frame"]
            if item is None or item is last:
                time.sleep(0.05)
                continue
            t0 = time.time()
            idle = t0 - alpr.motion_time > 2.0 and t0 - last_run < IDLE_VEHICLE_INTERVAL and not traffic.active(t0)
            if idle and last is not None and last[2] == item[2]:
                time.sleep(0.1)
                continue
            if last is not None and last[2] != item[2]:
                traffic.reset()
            last, last_run = item, t0
            try:
                objs = detect_objects(item[1])
                people = [o for o in objs if o.label == "person"]
                traffic.update(objs, t0, item[1].shape[1], item[1])
                state["det"] = (item[0], item[2], objs)
                counter.update(len(people))
            except Exception as e:
                print(f"Bílagreining mistókst: {e}")
                time.sleep(5)
            moving = time.time() - alpr.motion_time < 2.0 or traffic.active(time.time())
            time.sleep(max(0.0, (MOVING_VEHICLE_INTERVAL if moving else VEHICLE_INTERVAL) - (time.time() - t0)))

    state["frame"], state["det"], state["plates"], state["watch"] = None, None, None, 0.0
    tracker, plate_tracker = BoxTracker(), BoxTracker(shrink=0)
    counter, traffic = PeopleCounter(), VehicleTracks()
    night = NightCounter(traffic._count)
    seen_det = [None]
    motion = MotionTracker(COUNT_LINE)
    motion.on_cross = lambda direction, mid: traffic._count(motion.label_of(mid), direction, "motion")

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=plate_loop, daemon=True).start()
    threading.Thread(target=vehicle_loop, daemon=True).start()

    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    try:
        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
    except OSError:
        if not window:
            raise
        server = None

    if not window:
        print(f"Numby ALPR {VERSION} í gangi: {url}  (Ctrl+C til að hætta)")
        if host == "127.0.0.1":
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            counter.flush()
            cap.release()
        return

    # --window: sama mynd og vefviðmótið sýnir, í OpenCV-glugga með hliðarspjaldi
    if server is not None:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Numby ALPR {VERSION} í glugga (q til að hætta)." + (f" Vefviðmótið er líka opið á {url}" if server else ""))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    sized, thumbs, shown = False, {}, None
    try:
        while True:
            state["watch"] = time.time()
            view = state.get("view")
            if view is not None and view is not shown:
                shown = view
                h, w = view.shape[:2]
                if h > DISPLAY_MAX_H:
                    view = cv2.resize(view, (int(w * DISPLAY_MAX_H / h), DISPLAY_MAX_H), interpolation=cv2.INTER_AREA)
                info = f"Bílar í dag: {traffic.left + traffic.right} (< {traffic.left}  {traffic.right} >)   Fólk: {counter.now}"
                with lock:
                    panel = build_panel(view.shape[0], logs, thumbs, info)
                out = np.hstack([view, panel])
                if not sized:
                    cv2.resizeWindow(WINDOW, out.shape[1], out.shape[0])
                    sized = True
                cv2.imshow(WINDOW, out)
            if cv2.waitKey(15) & 0xFF == ord("q"):
                break
            try:
                if sized and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        pass
    finally:
        counter.flush()
        cap.release()
        cv2.destroyAllWindows()


def run(source: str | int = 0) -> None:
    """OpenCV-gluggi (--window)."""
    dashboard(source, window=True)
