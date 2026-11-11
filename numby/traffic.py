"""Bílatalning, rakning milli ramma og ljósatalning í myrkri."""
from datetime import datetime
import os
import threading
import time
from math import hypot
from types import SimpleNamespace as NS

from .colors import appearance, color_name, scene_cast, similar, APPEAR_MIN, APPEAR_PARKED_MIN, APPEAR_WEIGHT
from .objects import VEHICLE_LABELS

TRAFFIC_FILE = "traffic.csv"   # einn bíll á línu: tími, tegund og stefna (engar myndir)
COUNT_LINE = 0.75              # talningarlína: lóðrétt lína við þetta hlutfall af myndbreidd (0 = vinstri brún)
OCCLUSION_SECONDS = 4.0        # bíll á ferð sem hverfur (t.d. bak við annan) er rakinn áfram þetta lengi
PARKED_KEEP_SECONDS = 10.0     # kyrrstæður bíll sem hverfur (bíll keyrir fyrir framan hann) er geymdur þetta lengi
TRACK_NEW_CONF = 0.5           # óöruggari greining má fylgja bíl sem þegar er rakinn, en stofnar ekki nýjan
COUNT_DEDUPE_SECONDS = 1.5     # bílagreining og hreyfirakning telja sama bíl: seinni talningin innan þessa tíma er hunsuð
MOVING_VEHICLE_INTERVAL = 0.3  # bílagreining keyrir örar (3x á sekúndu) meðan eitthvað hreyfist

VEHICLE_NAMES = {
    "ljós": "Bíll (ljós í myrkri)",
    "motion": "Farartæki",
    "car": "Bíll",
    "truck": "Vörubíll",
    "bus": "Rúta",
    "motorcycle": "Mótorhjól",
}


class VehicleTracks:
    """Gefur hverjum bíl númer (#1, #2, ...) milli greininga og telur þá sem keyra framhjá.

    Hver bíll hefur staðsetningu og hraða. Við nýja greiningu er spáð hvar hver bíll ætti að vera og
    næsti kassi við spána tekinn. Bíll sem finnst ekki (t.d. bak við annan bíl) heldur áfram á sama
    hraða í allt að OCCLUSION_SECONDS og fær sama númer ef hann birtist aftur nálægt spánni.
    Bíll telst þegar hann keyrir yfir talningarlínuna (COUNT_LINE). Kyrrstæðir bílar eru aldrei taldir."""

    def __init__(self, path: str = TRAFFIC_FILE):
        self.path, self.tracks, self.next_id, self.frame_w, self.night = path, [], 1, 1920, False
        self._lock = threading.Lock()
        self._recent = []  # (tími, stefna, uppruni) síðustu talninga, til að telja ekki sama bíl tvisvar
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.left, self.right = 0, 0
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                next(f, None)
                for line in f:
                    p = line.strip().split(",")
                    if len(p) == 4 and p[0] == self.today:
                        self.left += (p[3] == "vinstri")
                        self.right += (p[3] == "hægri")

    def _check_day(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.today:
            self.today, self.left, self.right = today, 0, 0

    def reset(self) -> None:
        self.tracks = []  # t.d. við breyttan aðdrátt: gömul hnit gilda ekki

    def update(self, objs, now: float, frame_w: int, frame=None) -> None:
        """objs = niðurstöður detect_objects. Bílar fá eigindið .tid (númer bílsins).
        Með frame er litur hvers bíls notaður til að þekkja hann aftur eftir að hann hverfur bak við annað."""
        self._check_day()
        self.frame_w = frame_w
        dets = []
        cast = scene_cast(frame) if frame is not None else None
        for o in objs:
            if o.label in VEHICLE_LABELS:
                b = o.bounding_box
                o.app = appearance(frame, b) if frame is not None else None
                o.color = color_name(frame, b, cast) if frame is not None else None
                dets.append(((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, b.x2 - b.x1, b.y2 - b.y1, o))
        MAX_SPEED_PX_S = 1.2 * frame_w
        pairs = []
        for ti, t in enumerate(self.tracks):
            dt = now - t["t"]
            px, py = t["cx"] + t["vx"] * dt, t["cy"] + t["vy"] * dt
            # Fyrir nýjan bíl (hits=1) vitum við ekki hraðann enn: gefa svigrúm fyrir bíl á mikilli ferð
            gate_x = 0.6 * t["w"] + 0.5 * abs(t["vx"]) * dt
            gate_y = 0.6 * t["h"] + 0.5 * abs(t["vy"]) * dt
            for di, d in enumerate(dets):
                dx = abs(d[0] - px)
                dy = abs(d[1] - py)
                dist, ratio = hypot(dx, dy), (d[2] * d[3]) / max(1.0, t["w"] * t["h"])
                # Kyrrstæður bíll má ekki stela stórum hlut (t.d. rútu) sem keyrir framhjá
                parked = t["hits"] >= 3 and not self._moving(t)
                if parked:
                    if ratio > 1.8 or ratio < 0.5:
                        continue
                    if t["votes"] and max(t["votes"], key=t["votes"].get) == "car" and d[4].label in ("bus", "truck"):
                        continue
                # Bíll á staðfestri ferð í eina átt má ekki stökkva aftur á bak (gegn akstursstefnu)
                if self._moving(t) and t["hits"] >= 3 and abs(t["vx"]) > 0.02 * frame_w:
                    step_dx = d[0] - t["cx"]
                    if t["vx"] > 0 and step_dx < -0.3 * t["w"]:
                        continue
                    if t["vx"] < 0 and step_dx > 0.3 * t["w"]:
                        continue
                if dx <= gate_x and dy <= gate_y and (0.05 if self._moving(t) else 0.2) <= ratio <= 5:
                    sim = similar(t["app"], d[4].app)
                    if sim >= (APPEAR_PARKED_MIN if parked else APPEAR_MIN):
                        pairs.append(((dx / gate_x) + (dy / gate_y) + APPEAR_WEIGHT * (1 - sim), ti, di, ratio))
        used_t, used_d = set(), set()
        for dist, ti, di, ratio in sorted(pairs):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, (cx, cy, w, h, o) = self.tracks[ti], dets[di]
            t["votes"][o.label] = t["votes"].get(o.label, 0) + o.confidence
            # tegundin er sú sem hefur oftast sést, svo merkið flökti ekki milli 'Bíll' og 'Vörubíll'
            o.tid, t["seen"], o.label = t["id"], now, max(t["votes"], key=t["votes"].get)
            o.moving = self._moving(t)
            seen_color = o.color
            o.color = max(t["colors"], key=t["colors"].get) if t["colors"] else seen_color
            x1, x2 = cx - w / 2, cx + w / 2
            # Hraðinn er lesinn af brúnum kassans. Brún sem er falin (bak við annan bíl eða utan myndar)
            # stendur kyrr þótt bíllinn hreyfist, svo ef brúnunum ber ekki saman gildir sú sem hreyfist hraðar.
            de = now - t["et"]
            if de > 0:
                vl, vr = (x1 - t["ex1"]) / de, (x2 - t["ex2"]) / de
                agree = abs(vl - vr) <= max(0.3 * max(abs(vl), abs(vr)), 0.01 * frame_w)
                if agree:
                    vx = (vl + vr) / 2
                elif t["hits"] >= 3 and abs(t["vx"]) > 0.01 * frame_w:
                    vx = min(vl, vr, key=lambda v: abs(v - t["vx"]))  # þekktur hraði: trúa brúninni sem passar við hann
                else:
                    vx = max(vl, vr, key=abs)  # nýr bíll: falda brúnin stendur kyrr, hin sýnir hraðann
                k = 0.5 if t["hits"] > 1 or t["vx"] else 1.0
                if t["hits"] >= 3 and abs(vx - t["vx"]) > max(abs(t["vx"]), 0.05 * frame_w):
                    # ótrúlegt stökk í hraða: kassinn er líklega annar bútur af sama bíl hinum megin við
                    # það sem skyggir á (t.d. framendi vörubíls eftir að afturendinn sást). Hraðinn stendur,
                    # nema næsta mæling staðfesti stökkið (bíll sem tekur í alvöru af stað).
                    rej, t["vrej"] = t.get("vrej"), vx
                    if rej is None or abs(vx - rej) > max(0.5 * abs(vx), 0.03 * frame_w):
                        vx, k = t["vx"], 0.0
                else:
                    t["vrej"] = None
                if t["hits"] >= 6 and not self._moving(t):
                    # bíll sem hefur lengi staðið kyrr: kassinn hans bólgnar stundum þegar annar keyrir
                    # framhjá. Hann má ekki 'fara af stað' á því og stela númerinu af hinum.
                    vx, k = (vl + vr) / 2, 0.2
                dt = now - t["t"]  # færa spána fram til núna áður en hraðinn breytist
                t["cx"], t["cy"], t["t"] = t["cx"] + t["vx"] * dt, t["cy"] + t["vy"] * dt, now
                t["vx"] = t["vx"] * (1 - k) + vx * k
                if t["full"] and ratio >= 0.8:  # lóðréttur hraði aðeins milli tveggja heilla kassa
                    t["vy"] = t["vy"] * (1 - k) + (cy - t["ecy"]) / de * k
            o.vx = float(t.get("vx", 0.0))
            o.vy = float(t.get("vy", 0.0))
            t.update(ex1=x1, ex2=x2, ecy=cy, et=now, full=ratio >= 0.8)
            if not t["full"]:
                continue  # kassinn snarminnkaði = hálffalinn: staðsetningin kemur úr spánni, ekki miðju kassans
            prev_cx = t["lcx"]  # síðasta staðsetning sem sást í heilu lagi
            t.update(cx=cx, cy=cy, lcx=cx, w=w, h=h, t=now, hits=t["hits"] + 1)
            if seen_color:  # liturinn er sá sem oftast hefur sést þegar bíllinn sást allur
                t["colors"][seen_color] = t["colors"].get(seen_color, 0) + 1
                o.color = max(t["colors"], key=t["colors"].get)
            if o.app is not None:  # fingrafarið uppfærist hægt, og aðeins þegar bíllinn sést allur
                t["app"] = o.app if t["app"] is None else 0.8 * t["app"] + 0.2 * o.app
            # Talið þegar miðja bílsins fer yfir talningarlínuna. Hver bíll fer aðeins einu sinni yfir hana,
            # svo númeravíxl annars staðar í myndinni (t.d. bak við kyrrstæða bíla) valda ekki tvítalningu.
            line = COUNT_LINE * frame_w
            crossed = (prev_cx - line) * (cx - line) <= 0 and prev_cx != cx
            if not self.night and not t["counted"] and crossed and self._moving(t) \
                    and (cx - prev_cx) * t["vx"] > 0 and abs(t["vx"]) > 0.02 * frame_w:
                t["counted"] = True
                self._count(o.label, "hægri" if cx > prev_cx else "vinstri")
        for di, (cx, cy, w, h, o) in enumerate(dets):
            if di not in used_d and o.confidence >= TRACK_NEW_CONF:
                o.tid, o.moving = self.next_id, False
                o.vx, o.vy = 0.0, 0.0
                self.tracks.append({
                    "id": self.next_id, "cx": cx, "cy": cy, "w": w, "h": h, "vx": 0.0, "vy": 0.0,
                    "ex1": cx - w / 2, "ex2": cx + w / 2, "ecy": cy, "et": now, "full": True, "t": now,
                    "seen": now, "x0": cx, "hits": 1, "counted": False, "votes": {o.label: o.confidence},
                    "lcx": cx, "app": o.app,
                    "colors": {o.color: 1} if o.color else {}
                })
                self.next_id += 1
            elif not hasattr(o, "vx"):
                o.vx, o.vy = 0.0, 0.0
        self.tracks = [t for t in self.tracks
                       if now - t["seen"] <= (OCCLUSION_SECONDS if self._moving(t) else PARKED_KEEP_SECONDS)]

    def active(self, now: float) -> bool:
        """Er einhver bíll á ferð? Þá þarf að greina oft, þótt hreyfingin sé lítil í dílum (fjarlæg gata)."""
        return any(now - t["seen"] < 3.0 and self._moving(t) and abs(t["vx"]) > 0.01 * self.frame_w for t in self.tracks)

    def _moving(self, t: dict) -> bool:
        """Bíll sem hefur í alvöru færst úr stað (ekki bara titringur í kassa kyrrstæðs bíls)."""
        return t["counted"] or abs(t["cx"] - t["x0"]) >= 0.03 * self.frame_w or (t.get("hits", 0) >= 2 and abs(t.get("vx", 0.0)) > 0.015 * self.frame_w)

    def hidden(self, now: float) -> list:
        """Bílar á ferð sem sjást ekki núna (spáð staðsetning), til að teikna strikaðan kassa."""
        out = []
        for t in self.tracks:
            dt = now - t["seen"]
            if dt > 0.75 and self._moving(t):
                e = now - t["t"]
                cx, cy = t["cx"] + t["vx"] * e, t["cy"] + t["vy"] * e
                out.append(NS(tid=t["id"], bounding_box=NS(x1=cx - t["w"] / 2, y1=cy - t["h"] / 2,
                                                             x2=cx + t["w"] / 2, y2=cy + t["h"] / 2)))
        return out

    def _count(self, label: str, direction: str, source: str = "det") -> bool:
        """Skráir einn bíl. source segir hver taldi: "det" (bílagreining/ljós) eða "motion" (hreyfirakning).
        Ef hinn uppruninn taldi bíl í sömu átt rétt áður er þetta sami bíllinn og talningin hunsuð."""
        with self._lock:
            mono = time.time()
            self._recent = [r for r in self._recent if mono - r[0] <= COUNT_DEDUPE_SECONDS]
            if any(r[1] == direction and r[2] != source for r in self._recent):
                return False
            self._recent.append((mono, direction, source))
            now = datetime.now()
            day = now.strftime("%Y-%m-%d")
            if day != self.today:
                self.today, self.left, self.right = day, 0, 0
            self.left += (direction == "vinstri")
            self.right += (direction == "hægri")
            new = not os.path.isfile(self.path)
            with open(self.path, "a", encoding="utf-8") as f:
                if new:
                    f.write("Dagur,Tími,Tegund,Stefna\n")
                f.write(f"{day},{now.strftime('%H:%M:%S')},{VEHICLE_NAMES.get(label, label)},{direction}\n")
        return True


NIGHT_DARK, NIGHT_LIGHT = 100, 140  # miðgildi birtu í ræmunni við talningarlínuna (0-255): undir = nótt, yfir = dagur
NIGHT_MIN_SPEED = 0.06            # ljós þarf að fara þetta hlutfall af myndbreidd á sekúndu til að teljast bíll
NIGHT_MERGE_SECONDS = 0.7         # fram- og afturljós sama bíls fara yfir línuna með minna en þessu millibili


class NightCounter:
    """Telur bíla í myrkri eftir ljósunum. Þá sér bílagreiningin ekkert á götunni, en fram- og afturljós
    sjást sem bjartir blettir sem renna yfir myndina. Fylgst er með mjórri ræmu í kringum talningarlínuna:
    bjartir blettir sem hreyfast lárétt eru raktir ramma fyrir ramma og taldir þegar þeir fara yfir línuna.
    Ljós sama bíls (fram- og afturljós) eru sameinuð í eina talningu."""

    W = 640  # myndin er smækkuð í þessa breidd; öll mál hér að neðan miðast við hana

    def __init__(self, on_count):
        self.on_count, self.bg, self.tracks, self.events = on_count, None, [], []
        self.night, self.shape, self.last_t, self.warm, self.checked = False, None, 0.0, 0.0, -9.0

    def update(self, frame, now: float) -> bool:
        """Kallað á hverjum ramma. Skilar True ef nótt er (og ljósatalning því virk)."""
        import cv2
        import numpy as np
        if not self.night and now - self.checked < 1.0:
            return False  # að degi til er aðeins gáð einu sinni á sekúndu hvort orðið sé dimmt
        h, w = frame.shape[:2]
        k = self.W / w
        line, half = int(COUNT_LINE * self.W), int(0.15 * self.W)
        x0, x1 = max(0, int((line - half) / k)), min(w, int((line + half) / k))
        roi = cv2.resize(frame[:, x0:x1], (max(2, int((x1 - x0) * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA)
        v = roi.max(axis=2).astype(np.float32)  # bjartasta litrás: rauð afturljós sjást jafn vel og hvít framljós
        dt = now - self.last_t
        self.last_t = now
        if self.bg is None or self.shape != (h, w) or dt > 2.0:
            self.bg, self.shape, self.tracks, self.warm = v, (h, w), [], now + 1.5
            return self.night
        if now - self.checked >= 1.0:
            self.checked = now
            was = self.night
            self.night = float(np.median(v)) < (NIGHT_LIGHT if self.night else NIGHT_DARK)
            if self.night != was:
                self.bg = None  # byrja upp á nýtt þegar skipt er milli dags og nætur
                print("Myrkur: bílar eru nú taldir eftir ljósum" if self.night else "Bjart: bílar eru nú taldir með bílagreiningu")
                return self.night
        if not self.night:
            return False
        fg = ((v - self.bg) > 35).astype(np.uint8)
        self.bg += 0.03 * (v - self.bg)  # bakgrunnur: kyrr ljós (staurar, speglun) hverfa úr forgrunni
        n, _, stats, cent = cv2.connectedComponentsWithStats(cv2.dilate(fg, np.ones((1, 5), np.uint8)))
        lx = line - int(x0 * k)  # talningarlínan í hnitum ræmunnar
        blobs = [(cent[i][0], cent[i][1]) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= 4 and stats[i, cv2.CC_STAT_HEIGHT] <= 12]
        max_step = 1.6 * self.W * max(dt, 0.01)  # hraðasti bíll: 1,6 myndbreiddir á sekúndu
        used = set()
        for t in self.tracks:
            best = None
            for bi, (x, y) in enumerate(blobs):
                dx = x - t["x"]
                if bi in used or abs(y - t["y"]) > 4 or not 1.0 <= abs(dx) <= max_step:
                    continue
                if t["n"] >= 2 and dx * t["dir"] <= 0:
                    continue  # ljós snýr ekki við
                if best is None or abs(dx) < abs(blobs[best][0] - t["x"]):
                    best = bi
            if best is None:
                continue
            used.add(best)
            x, y = blobs[best]
            crossed = (t["x"] - lx) * (x - lx) <= 0
            t.update(x=x, y=y, dir=1 if x > t["x"] else -1, n=t["n"] + 1, seen=now)
            speed = abs(x - t["x0"]) / max(now - t["t0"], 1e-3)
            if crossed and not t["done"] and t["n"] >= 3 and speed >= NIGHT_MIN_SPEED * self.W and now >= self.warm:
                t["done"] = True
                self._event(now, t["dir"], y)
        self.tracks = [t for t in self.tracks if now - t["seen"] <= 0.25]
        for bi, (x, y) in enumerate(blobs):
            if bi not in used:
                self.tracks.append({"x": x, "y": y, "x0": x, "t0": now, "dir": 0, "n": 1, "seen": now, "done": False})
        return True

    def _event(self, now: float, direction: int, y: float) -> None:
        self.events = [e for e in self.events if now - e[0] <= NIGHT_MERGE_SECONDS]
        for e in self.events:
            if e[1] == direction and abs(e[2] - y) <= 40:
                e[0] = now  # annað ljós á sama bíl (eða eftirvagn): ekki nýr bíll
                return
        self.events.append([now, direction, y])
        self.on_count("ljós", "hægri" if direction > 0 else "vinstri")
