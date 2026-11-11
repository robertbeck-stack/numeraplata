"""Flísalögð plötugreining með fast-alpr og hreyfiskynjun."""
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import cv2
import numpy as np

from .plates import valid_plate

SCAN_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "global-plates-mobile-vit-v2-model"
MOTION_DIFF = 10        # birtubreyting díls (0-255) sem telst hreyfing
MOTION_MIN = 0.002      # hlutfall breyttra díla í flís svo hún sé skönnuð strax
HOT_SECONDS = 1.0       # flís þar sem plata sást er skönnuð áfram þetta lengi
MIN_PLATE_PX = 44       # mjórri plötur eru ólæsilegar: OCR skilar bulli, svo þeim er sleppt
SCAN_SESSIONS = 4       # mest svona mörg svæði greind samhliða (2 þræðir hvert)
MIN_SCAN_PERIOD = 0.05  # mest 20 skannanir á sekúndu
IDLE_SCAN_PERIOD = 0.2  # þegar ekkert hreyfist og engin plata sést: 5 á sekúndu (sparar örgjörva)
IDLE_VEHICLE_INTERVAL = 1.0  # bíla-/fólksgreining keyrir í síðasta lagi á 1 sekúndu millibili


def _raise_thread_priority() -> None:
    """Hækkar forgang þráðarins sem les myndavélina, svo greiningarnar (sem nota marga kjarna)
    svelti ekki lifandi myndina þegar vélin er undir álagi. Aðeins á Windows."""
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            k32.SetThreadPriority(k32.GetCurrentThread(), 1)  # THREAD_PRIORITY_ABOVE_NORMAL
        except Exception:
            pass


def _ort_opts(threads: int):
    """ONNX Runtime stillingar: fastur kjarnafjöldi og ekkert 'spinning', svo líkönin
    (plötur og bílar) keppi ekki um örgjörvann þegar þau keyra samhliða."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, threads)
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


class PlateScanner:
    """Plötugreining í flísum. Skynjarinn minnkar allt sem hann fær niður í 384 px, svo í heilli
    1080p mynd þarf plata að vera ~130 px breið til að finnast. Hér er myndinni líka skipt í allt að
    3x2 flísar sem skarast, og þá duga ~55 px (bíllinn má vera meira en tvöfalt lengra í burtu).

    Til að það kosti ekki 7 greiningar á ramma eru aðeins skannaðar flísar þar sem eitthvað hreyfist
    eða plata sást nýlega, plús ein flís í röð í hverri umferð svo kyrrstæðir bílar finnist líka.
    Flísarnar keyra samhliða, hver á sínu eintaki af skynjaranum."""

    def __init__(self):
        from fast_alpr.default_detector import DefaultDetector
        from fast_alpr.default_ocr import DefaultOCR
        cores = os.cpu_count() or 4
        # mörg lítil eintök samhliða nýta kjarnana mun betur en eitt eintak með marga þræði (mælt)
        n = min(SCAN_SESSIONS, max(1, (cores - 4) // 2))
        self.free = queue.LifoQueue()  # síðast notaða eintakið fyrst: það er 'heitt'
        for _ in range(n):
            self.free.put(DefaultDetector(SCAN_MODEL, conf_thresh=0.4, sess_options=_ort_opts(2)))
        self.ocr = DefaultOCR(OCR_MODEL, sess_options=_ort_opts(2))
        self.pool = ThreadPoolExecutor(n)
        self.prev, self.hot, self.rr, self.moving = None, {}, 0, set()
        self.busy, self.motion_time = True, time.time()  # lesið af öðrum þráðum

    @staticmethod
    def regions(w: int, h: int) -> list[tuple]:
        """Svæði (x, y, breidd, hæð): heildarmyndin fyrst (stórar plötur), svo flísar (litlar plötur)."""
        nx, ny = min(3, max(1, round(w / 640))), min(2, max(1, round(h / 540)))
        out = [(0, 0, w, h)]
        if nx * ny > 1:
            tw = w if nx == 1 else int(w / nx * 1.2)
            th = h if ny == 1 else int(h / ny * 1.2)
            out += [(i * (w - tw) // max(1, nx - 1), j * (h - th) // max(1, ny - 1), tw, th)
                    for j in range(ny) for i in range(nx)]
        return out

    def _active(self, frame, regs, now: float) -> set:
        h, w = frame.shape[:2]
        k = 320 / w
        small = cv2.resize(frame, (320, max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        prev, self.prev = self.prev, g
        if prev is None or prev.shape != g.shape:
            self.busy, self.motion_time = True, now
            self.moving = set(range(len(regs)))
            return set(self.moving)
        moved = cv2.absdiff(g, prev) > MOTION_DIFF
        # heildarmyndin (0) er ekki skönnuð við hreyfingu: flísarnar sjá það sama í hærri upplausn.
        # Hún kemur inn í röðinni, og strax ef flís sér plötu sem nær út fyrir flísina (stór plata).
        act = {i for i, (x, y, tw, th) in enumerate(regs)
               if (i or len(regs) == 1) and moved[int(y * k):int((y + th) * k) + 1, int(x * k):int((x + tw) * k) + 1].mean() >= MOTION_MIN}
        self.moving = set(act)
        if moved.mean() >= MOTION_MIN / 6:  # einhver hreyfing í myndinni (t.d. ein flís af sex)
            self.motion_time = now
        act |= {i for i, t in self.hot.items() if now - t < HOT_SECONDS and i < len(regs)}
        self.busy = bool(act)
        self.rr = (self.rr + 1) % len(regs)
        act.add(self.rr)
        return act

    def predict(self, frame, everything: bool = False) -> list:
        """Skilar lista af niðurstöðum með .detection.bounding_box og .ocr (eins og fast_alpr.ALPR).
        everything=True skannar öll svæði (stakar myndir, þar sem hreyfing segir ekkert)."""
        h, w = frame.shape[:2]
        now = time.time()
        regs = self.regions(w, h)
        act = set(range(len(regs))) if everything else self._active(frame, regs, now)
        # 'heitt' svæði = plata sást þar meðan eitthvað hreyfðist; fylgir bíl sem hægir á sér eða stoppar.
        # Kyrrstæðir bílar halda svæði ekki heitu (þá væri það skannað endalaust), röðin sér um þá.
        moving = act if everything else self.moving

        def job(i):
            x, y, tw, th = regs[i]
            det = self.free.get()
            try:
                found = det.predict(frame if i == 0 else np.ascontiguousarray(frame[y:y + th, x:x + tw]))
            finally:
                self.free.put(det)
            out = []
            for d in found:
                b = d.bounding_box
                if b.x2 - b.x1 < MIN_PLATE_PX:
                    continue
                # plata sem snertir innri brún flísar er afklippt; næsta flís eða heildarmyndin sér hana heila
                if i and ((b.x1 <= 2 and x > 0) or (b.y1 <= 2 and y > 0)
                          or (b.x2 >= tw - 2 and x + tw < w) or (b.y2 >= th - 2 and y + th < h)):
                    if i in moving:
                        self.hot[0] = now
                    continue
                out.append((i > 0, d.confidence, max(0, b.x1 + x), max(0, b.y1 + y), min(w, b.x2 + x), min(h, b.y2 + y), i))
            return out

        found = sorted((f for res in self.pool.map(job, sorted(act)) for f in res), reverse=True)
        kept = []
        for f in found:  # raðað: flísar á undan heildarmynd (nákvæmari kassi), svo eftir öryggi
            if f[6] in moving:
                self.hot[f[6]] = now
            for q, alts in kept:  # sama platan úr tveimur svæðum
                ix, iy = min(f[4], q[4]) - max(f[2], q[2]), min(f[5], q[5]) - max(f[3], q[3])
                if ix > 0 and iy > 0 and ix * iy > 0.4 * min((f[4] - f[2]) * (f[5] - f[3]), (q[4] - q[2]) * (q[5] - q[3])):
                    alts.append(f)
                    break
            else:
                kept.append((f, []))
        results = []
        for f, alts in kept:
            best = None
            # ef besti kassinn gefur ekki gilt númer er annar kassi af sömu plötu prófaður (önnur útklippa)
            for _, conf, x1, y1, x2, y2, _ in [f] + alts[:2]:
                crop = frame[y1:y2, x1:x2]
                res = self.ocr.predict(crop) if crop.size else None
                if res is None:
                    continue
                ok = bool(valid_plate(res.text))
                if best is None or ok:
                    best = NS(detection=NS(bounding_box=NS(x1=x1, y1=y1, x2=x2, y2=y2), confidence=conf), ocr=res)
                if ok:
                    break
            if best is not None:
                results.append(best)
        return results


def make_alpr():
    return PlateScanner()
