"""Minimal ALPR: CLI webcam/stream OR mobile web app (camera + timestamp log)."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import re
import sys
import cv2
import numpy as np
from fast_alpr import ALPR

PLATE_RE = re.compile(r"^[A-Z0-9]{4,8}$")
LOG_FILE = "plates.csv"

HTML = b"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Numby ALPR</title>
<style>body{font:16px system-ui;margin:0;background:#111;color:#eee;text-align:center}
video{width:100%;max-height:45vh;background:#000}table{width:100%;border-collapse:collapse}
td,th{padding:8px;border-bottom:1px solid #333;text-align:left}</style>
<video id=v autoplay playsinline></video><canvas id=c hidden></canvas>
<table><thead><tr><th>T\xc3\xadmi</th><th>B\xc3\xadln\xc3\xbamer</th><th>%</th></tr></thead><tbody id=log></tbody></table>
<script>
const v=document.getElementById('v'),c=document.getElementById('c'),log=document.getElementById('log');
navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}}).then(s=>v.srcObject=s);
setInterval(()=>{
  if(!v.videoWidth)return;
  c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(b=>fetch('/scan',{method:'POST',body:b}).then(r=>r.json()).then(rows=>{
    rows.forEach(x=>log.insertAdjacentHTML('afterbegin',`<tr><td>${x.time}</td><td><b>${x.plate}</b></td><td>${x.conf}%</td></tr>`));
  }),'image/jpeg',0.8);
},1000);
</script>"""


def valid_plate(text: str) -> str | None:
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleaned if PLATE_RE.match(cleaned) else None


def make_alpr() -> ALPR:
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="global-plates-mobile-vit-v2-model",
    )


def extract_new_plates(alpr: ALPR, frame: np.ndarray, seen: set[str]) -> list[dict]:
    found = []
    for res in alpr.predict(frame):
        if res.ocr and res.ocr.text and res.ocr.confidence > 0.7:
            plate = valid_plate(res.ocr.text)
            if plate and plate not in seen:
                seen.add(plate)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conf = round(res.ocr.confidence * 100)
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{ts},{plate},{conf}\n")
                found.append({"time": ts, "plate": plate, "conf": conf})
    return found


def serve(port: int = 8000) -> None:
    # ponytail: single-threaded stdlib HTTPServer; switch to FastAPI/uvicorn if multi-user
    alpr, seen = make_alpr(), set()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML)

        def do_POST(self):
            buf = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            hits = extract_new_plates(alpr, frame, seen) if frame is not None else []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(hits).encode())

        def log_message(self, *args):
            pass

    print(f"Open on Android Chrome: http://localhost:{port} (or via LAN IP + HTTPS/adb forward)")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def run(source: str | int = 0) -> None:
    alpr, seen, frame_idx = make_alpr(), set(), 0
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % 5 == 0:
            for hit in extract_new_plates(alpr, frame, seen):
                print(f"{hit['time']}\t{hit['plate']}\t({hit['conf']}%)")
            frame = alpr.draw_predictions(frame)
        cv2.imshow("ALPR (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    assert valid_plate("AB-123") == "AB123"
    assert valid_plate("x") is None
    if "--test" in sys.argv:
        print("self-check ok")
    elif "--web" in sys.argv:
        serve()
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else 0)

