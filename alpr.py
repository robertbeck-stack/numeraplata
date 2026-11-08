"""Numby ALPR (Python): Íslenskur númeraplötulesari fyrir vefmyndavél, RTSP/símastraum og vefviðmót."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import re
import sys
import time

PLATE_RE = re.compile(r"^([A-Z]{2}[0-9]{3}|[A-Z]{3}[0-9]{2})$")
STICKER_RE = re.compile(r"^([A-Z0-9]{2})[0-9]{2}([A-Z0-9]{3})$")
LOG_FILE = "plates.csv"
CAPTURES_DIR = "captures"

TO_LET = {"0": "O", "1": "I", "8": "B"}
TO_DIG = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "B": "8", "S": "5"}


def _normalize_five(s: str) -> str:
    if len(s) != 5:
        return s
    if not (s[0].isalpha() or s[1].isalpha()):
        return s
    if not (s[3].isdigit() or s[4].isdigit()):
        return s
    return (
        TO_LET.get(s[0], s[0])
        + TO_LET.get(s[1], s[1])
        + s[2]
        + TO_DIG.get(s[3], s[3])
        + TO_DIG.get(s[4], s[4])
    )


def valid_plate(raw: str) -> str | None:
    cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper()).removeprefix("IS")
    norm5 = _normalize_five(cleaned)
    if PLATE_RE.match(norm5):
        return norm5
    m = STICKER_RE.match(cleaned)
    if m:
        combined = _normalize_five(m.group(1) + m.group(2))
        if PLATE_RE.match(combined):
            return combined
    return None


def is_same_car(a: str, b: str) -> bool:
    if len(a) != 5 or len(b) != 5:
        return False
    if sum(x != y for x, y in zip(a, b)) <= 1:
        return True
    dp = [[0] * 6 for _ in range(6)]
    for i in range(1, 6):
        for j in range(1, 6):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[5][5] >= 4


HTML = b"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Numby ALPR (Python)</title>
<style>
body{font:16px system-ui;margin:0;background:#121212;color:#eee}
video{width:100%;max-height:42vh;background:#000;display:block;object-fit:cover}
.bar{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#1e1e1e}
button,a.btn{background:#bb86fc;color:#000;border:0;padding:8px 14px;border-radius:18px;font-weight:700;text-decoration:none;cursor:pointer}
.day{background:#262626;padding:6px 14px;font-weight:700;font-size:14px;display:flex;justify-content:space-between}
.row{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid #2a2a2a;cursor:pointer}
.row img{height:48px;width:92px;object-fit:cover;border-radius:4px}
dialog{background:#1e1e1e;color:#eee;border:1px solid #444;border-radius:12px;max-width:92vw;padding:16px}
dialog img{max-width:85vw;max-height:65vh;display:block;margin:10px 0}
</style>
<video id=v autoplay playsinline></video><canvas id=c hidden></canvas>
<div class=bar><b id=cnt>Atvik (0)</b><div><a class=btn href="/plates.csv" download="plates.csv">Deila CSV</a></div></div>
<div id=list></div>
<dialog id=dlg onclick="if(event.target===this)this.close()">
  <h3 id=d_title style="margin:0"></h3><small id=d_time></small>
  <img id=d_img><div style="display:flex;gap:8px;justify-content:flex-end">
  <a id=d_dl class=btn download>S\xc3\xa6kja mynd</a><button onclick="dlg.close()">Loka</button></div>
</dialog>
<script>
const v=document.getElementById('v'),c=document.getElementById('c'),list=document.getElementById('list'),cnt=document.getElementById('cnt'),dlg=document.getElementById('dlg');
navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1920},height:{ideal:1080}}}).then(s=>v.srcObject=s);
let events=[];
function render(){
  cnt.textContent=`Atvik (${events.length})`;
  const byDay={};
  events.forEach(e=>{(byDay[e.time.split(' ')[0]]??=[]).push(e)});
  list.innerHTML=Object.entries(byDay).map(([d,arr])=>
    `<div class=day><span>${d}</span><span>${arr.length} atvik</span></div>`+
    arr.map((x,i)=>`<div class=row onclick='show(${JSON.stringify(x)})'><div><span style="color:#bb86fc;margin-right:12px">${x.time.split(' ')[1]}</span><b style="font-size:18px">${x.plate}</b></div>${x.img?`<img src="${x.img}">`:''}</div>`).join('')
  ).join('');
}
function show(x){
  document.getElementById('d_title').textContent=x.plate;
  document.getElementById('d_time').textContent=x.time;
  const img=document.getElementById('d_img'),dl=document.getElementById('d_dl');
  if(x.img){img.src=x.img;img.hidden=false;dl.href=x.img;dl.hidden=false;}else{img.hidden=true;dl.hidden=true;}
  dlg.showModal();
}
setInterval(()=>{
  if(!v.videoWidth)return;
  c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(b=>fetch('/scan',{method:'POST',body:b}).then(r=>r.json()).then(all=>{
    if(all.length){events=all;render();}
  }),'image/jpeg',0.92);
},600);
</script>"""


def make_alpr():
    from fast_alpr import ALPR
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="global-plates-mobile-vit-v2-model",
    )


def _save_snapshot(frame, box, plate: str, ts: str) -> str:
    import cv2
    os.makedirs(CAPTURES_DIR, exist_ok=True)
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
    rel_path = f"{CAPTURES_DIR}/{fname}"
    cv2.imwrite(rel_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return "/" + rel_path


def process_frame(alpr, frame, logs: list[dict]) -> list[dict]:
    now = time.time()
    changed = False
    for res in alpr.predict(frame):
        if not (res.ocr and res.ocr.text and res.ocr.confidence >= 0.45):
            continue
        plate = valid_plate(res.ocr.text)
        if not plate:
            continue
        conf = round(res.ocr.confidence * 100)
        box = getattr(res.detection, "bounding_box", None)
        match_idx = next(
            (i for i, e in enumerate(logs) if now - e["ts_sec"] < 25.0 and (e["plate"] == plate or is_same_car(e["plate"], plate))),
            -1,
        )
        if match_idx == -1:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            img_url = _save_snapshot(frame, box, plate, ts)
            logs.insert(0, {"time": ts, "plate": plate, "conf": conf, "img": img_url, "ts_sec": now})
            changed = True
        elif conf > logs[match_idx]["conf"] and logs[match_idx]["plate"] != plate:
            old = logs[match_idx]
            old["plate"] = plate
            old["conf"] = conf
            old["img"] = _save_snapshot(frame, box, plate, old["time"])
            old["ts_sec"] = now
            changed = True

    if changed:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("Dagur,Tími,Bílnúmer,Öryggi,Mynd\n")
            for e in reversed(logs):
                day, clock = e["time"].split(" ")
                f.write(f"{day},{clock},{e['plate']},{e['conf']}%,{e['img']}\n")
    return logs if changed else []


def serve(port: int = 8000) -> None:
    import cv2
    import numpy as np
    alpr, logs = make_alpr(), []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/plates.csv" and os.path.exists(LOG_FILE):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.end_headers()
                with open(LOG_FILE, "rb") as f:
                    self.wfile.write(f.read())
                return
            if self.path.startswith("/captures/") and os.path.exists(self.path.lstrip("/")):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(self.path.lstrip("/"), "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML)

        def do_POST(self):
            buf = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            updated = process_frame(alpr, frame, logs) if frame is not None else []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = [{k: v for k, v in e.items() if k != "ts_sec"} for e in updated]
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    print(f"Vefviðmót í gangi: http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def run(source: str | int = 0) -> None:
    import cv2
    alpr, logs, frame_idx = make_alpr(), [], 0
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Gat ekki opnað myndavél/straum: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % 3 == 0:
            if process_frame(alpr, frame, logs):
                top = logs[0]
                print(f"{top['time']}\t{top['plate']}\t({top['conf']}%) -> {top['img']}")
            frame = alpr.draw_predictions(frame)
        cv2.imshow("Numby ALPR (q til að hætta)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    assert valid_plate("AB-123") == "AB123"
    assert valid_plate("IS LO 25 L46") == "LOL46"
    assert valid_plate("L0 25 L46") == "LOL46"
    assert valid_plate("AB 12O") == "AB120"
    assert valid_plate("PARIS") is None
    assert valid_plate("x") is None
    assert is_same_car("VSD07", "VSO07")
    assert is_same_car("XLS53", "IXL53")
    assert not is_same_car("VSD07", "ABC12")
    if "--test" in sys.argv:
        print("self-check ok")
    elif "--web" in sys.argv:
        serve()
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else 0)
