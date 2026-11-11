"""Hreyfirakning á hverjum ramma: fylgir því sem hreyfist án þess að bíða eftir bílagreiningunni."""
from collections import deque
from math import hypot
from types import SimpleNamespace as NS

import cv2
import numpy as np

MOTION_THRESHOLD = 12     # birtumunur (0-255) frá bakgrunni svo díll teljist hluti af hlut á hreyfingu
MOTION_MIN_AREA = 0.00015  # minnsta stærð hlutar sem hlutfall af myndfleti (síar burt suð og fugla)
MOTION_MIN_SPEED = 0.03   # minnsti hraði (myndbreiddir á sekúndu) svo hlutur teljist á ferð
MOTION_CONFIRM = 4        # rammar sem hlutur þarf að sjást á stöðugri hreyfingu áður en hann er sýndur
MOTION_MERGE_SECONDS = 0.6  # tveir hlutir yfir línuna í sömu átt með minna millibili = sami bíll (bíll + skuggi)


class MotionTracker:
    """Finnur og fylgir hlutum á hreyfingu með bakgrunnsfrádrætti. Þetta kostar um 2 ms á ramma og
    keyrir því á hverjum ramma, ólíkt bílagreiningunni sem tekur um hálfa sekúndu. Bíll sem fer yfir
    myndina á tveimur sekúndum fær þannig kassa sem fylgir honum strax, og hann er talinn þegar hann
    fer yfir talningarlínuna þótt bílagreiningin nái honum aldrei.

    Hreyfirakningin veit ekki HVAÐ hluturinn er. Tegund, litur og númer bílsins koma frá
    bílagreiningunni þegar hún nær honum (sjá dashboard)."""

    W = 480  # myndin er smækkuð í þessa breidd

    def __init__(self, count_line: float = 0.75, on_cross=None):
        self.count_line, self.on_cross = count_line, on_cross
        self.bg, self.shape, self.tracks, self.next_id = None, None, [], 1
        self.last_t, self.events = 0.0, []
        self.info = {}      # númer hreyfihlutar -> (tegund, öryggi, númer bíls, litur) frá bílagreiningunni
        self.claimed = {}   # númer bíls -> rammi þegar hann var síðast paraður við hreyfihlut
        self.det_seen = {}  # númer bíls -> í hve mörgum greiningum hann hefur sést

    def update(self, frame, now: float, frame_no: int = 0) -> list:
        """Kallað á hverjum ramma. Skilar hlutum á ferð: bounding_box (í hnitum rammans), vx/vy (dílar á
        ramma), mid (númer hreyfihlutar) og age (fjöldi ramma sem hann hefur sést)."""
        h, w = frame.shape[:2]
        k = self.W / w
        self.k = k
        sh = max(2, int(h * k))
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (self.W, sh), interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        dt = now - self.last_t
        self.last_t = now
        if self.bg is None or self.shape != (h, w) or dt > 1.0 or dt <= 0:
            self.bg, self.shape, self.tracks = g, (h, w), []
            return []
        fg = cv2.absdiff(g, self.bg) > MOTION_THRESHOLD
        if fg.mean() > 0.25:  # öll myndin breyttist (lýsing, myndavél hreyfð): byrja upp á nýtt
            self.bg, self.tracks = g, []
            return []
        # bakgrunnurinn lærist hratt þar sem ekkert hreyfist, hægt þar sem hlutur er (svo bíll sem stoppar
        # verði á endanum hluti af bakgrunninum, en bíll á ferð ekki)
        self.bg += np.where(fg, 0.002, 0.05).astype(np.float32) * (g - self.bg)
        mask = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 11), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
        min_area = MOTION_MIN_AREA * self.W * sh
        blobs = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area >= min_area and bh <= 0.5 * sh and bw <= 0.7 * self.W:
                blobs.append((cent[i][0], cent[i][1], x, y, bw, bh))

        used = set()
        for t in self.tracks:
            px, py = t["cx"] + t["vx"], t["cy"] + t["vy"]
            best, best_d = None, max(14.0, 0.9 * t["w"])  # leitarsvæði í kringum spána
            for bi, b in enumerate(blobs):
                d = hypot(b[0] - px, b[1] - py)
                if bi not in used and d < best_d and abs(b[1] - py) < max(8.0, 0.8 * t["h"]):
                    best, best_d = bi, d
            if best is None:
                t["miss"] += 1
                t["cx"], t["cy"] = px, py  # heldur áfram á sama hraða meðan hann sést ekki (t.d. bak við staur)
                continue
            used.add(best)
            cx, cy, x, y, bw, bh = blobs[best]
            vx, vy = cx - t["cx"], cy - t["cy"]
            a = 0.5 if t["n"] > 1 else 1.0
            prev_cx = t["cx"]
            t.update(cx=cx, cy=cy, x=x, y=y, w=bw, h=bh, vx=t["vx"] * (1 - a) + vx * a, vy=t["vy"] * (1 - a) + vy * a,
                     n=t["n"] + 1, miss=0)
            t["hist"].append((frame_no, x, y, bw, bh))
            self._check_cross(t, prev_cx, now, dt)
        self.tracks = [t for t in self.tracks if t["miss"] <= 6 and -20 < t["cx"] < self.W + 20]
        for bi, (cx, cy, x, y, bw, bh) in enumerate(blobs):
            if bi not in used:
                self.tracks.append({"id": self.next_id, "cx": cx, "cy": cy, "x0": cx, "y0": cy, "x": x, "y": y,
                                    "w": bw, "h": bh, "vx": 0.0, "vy": 0.0, "n": 1, "miss": 0, "done": False,
                                    "t0": now, "hist": deque([(frame_no, x, y, bw, bh)], maxlen=90)})
                self.next_id += 1

        out = []
        for t in self.tracks:
            if not self._confirmed(t, now) or t["miss"] > 2:
                continue
            bx = t["cx"] - t["w"] / 2  # kassinn fylgir miðjunni (líka meðan spáð er)
            by = t["cy"] - t["h"] / 2
            out.append(NS(mid=t["id"], age=t["n"], vx=t["vx"] / k, vy=t["vy"] / k,
                          bounding_box=NS(x1=bx / k, y1=by / k, x2=(bx + t["w"]) / k, y2=(by + t["h"]) / k)))
        return out

    def assign(self, det_objs: list, frame_no: int) -> None:
        """Parar niðurstöður bílagreiningar við hreyfihluti. Greiningin var gerð á ramma frame_no, sem er
        liðinn þegar hún berst, svo hún er borin saman við hvar hver hreyfihlutur VAR á þeim ramma.
        Hreyfihluturinn erfir tegund, öryggi, númer og lit."""
        k = getattr(self, "k", 1.0)
        info = self.info
        tids = {o.tid for o in det_objs if getattr(o, "tid", None) is not None}
        self.det_seen = {tid: self.det_seen.get(tid, 0) + 1 for tid in tids} | \
                        {tid: n for tid, n in self.det_seen.items() if tid not in tids and len(self.det_seen) < 400}
        self.claimed = {tid: f for tid, f in self.claimed.items() if frame_no - f < 600}
        for t in self.tracks:
            past = min(t["hist"], key=lambda e: abs(e[0] - frame_no))
            if abs(past[0] - frame_no) > 3:
                continue  # hluturinn var ekki til þá
            x1, y1, x2, y2 = past[1] / k, past[2] / k, (past[1] + past[3]) / k, (past[2] + past[4]) / k
            best, best_iou = None, 0.25
            for o in det_objs:
                b = o.bounding_box
                if getattr(o, "tid", None) is None or o.label == "person":
                    continue
                ix, iy = min(x2, b.x2) - max(x1, b.x1), min(y2, b.y2) - max(y1, b.y1)
                if ix <= 0 or iy <= 0:
                    continue
                iou = ix * iy / ((x2 - x1) * (y2 - y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - ix * iy)
                if iou > best_iou:
                    best, best_iou = o, iou
            if best is not None:
                info[t["id"]] = (best.label, best.confidence, best.tid, getattr(best, "color", None))
                self.claimed[best.tid] = frame_no
        alive = {t["id"] for t in self.tracks}
        self.info = {mid: v for mid, v in info.items() if mid in alive}

    def label_of(self, mid: int) -> str:
        return self.info[mid][0] if mid in self.info else "motion"

    def merge(self, vehicles: list, movers: list) -> tuple[list, list]:
        """Sameinar kassa bílagreiningarinnar (vehicles) og hreyfirakningarinnar (movers) fyrir teikningu.

        Hreyfikassinn er alltaf á réttum stað en veit ekki hvað hluturinn er; bílagreiningin veit það en
        kassinn hennar sýnir hvar bíllinn VAR fyrir hálfri sekúndu. Kassi bílagreiningarinnar er því aðeins
        teiknaður fyrir bíl sem stendur kyrr: hann þarf að hafa sést í tveimur greiningum og má ekki
        vera paraður við hreyfihlut. Annars sæti tómur kassi eftir þar sem bíll á ferð var.

        Skilar (kyrrir, á_ferð)."""
        still = [o for o in vehicles if o.tid not in self.claimed and self.det_seen.get(o.tid, 0) >= 2]
        moving = []
        for m in movers:
            label, conf, tid, color = self.info.get(m.mid, ("motion", None, None, None))
            moving.append(NS(label=label, confidence=conf, tid=tid, color=color, moving=True,
                             vx=m.vx, vy=m.vy, bounding_box=m.bounding_box))
        return still, moving

    def _confirmed(self, t: dict, now: float) -> bool:
        """Hlutur sem hefur hreyfst stöðugt í sömu átt nógu lengi (ekki suð, flökt eða laufblað)."""
        if t["n"] < MOTION_CONFIRM:
            return False
        moved = hypot(t["cx"] - t["x0"], t["cy"] - t["y0"])
        return moved >= 10.0 and moved / max(now - t["t0"], 1e-3) >= MOTION_MIN_SPEED * self.W

    def _check_cross(self, t: dict, prev_cx: float, now: float, dt: float) -> None:
        line = self.count_line * self.W
        if t["done"] or (prev_cx - line) * (t["cx"] - line) > 0 or not self._confirmed(t, now):
            return
        if abs(t["vx"]) < 2 * abs(t["vy"]):
            return  # fer aðallega upp eða niður: ekki umferð yfir línuna
        t["done"] = True
        direction = 1 if t["vx"] > 0 else -1
        self.events = [e for e in self.events if now - e[0] <= MOTION_MERGE_SECONDS]
        for e in self.events:
            if e[1] == direction and abs(e[2] - t["cy"]) <= 3 * max(t["h"], 8):
                e[0] = now  # bíll og skuggi hans, eða bíll sem klofnaði í tvo bletti
                return
        self.events.append([now, direction, t["cy"]])
        if self.on_cross:
            self.on_cross("hægri" if direction > 0 else "vinstri", t["id"])
