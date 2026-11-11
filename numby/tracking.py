"""Kassaraktari sem notar ljósflæði (Lucas-Kanade), hnitaspá og festingu við ökutæki."""
from collections import deque
from math import hypot
from types import SimpleNamespace
import cv2
import numpy as np


class BoxTracker:
    """Lætur kassa fylgja hlutum milli greininga með ljósflæði (Lucas-Kanade) og hraðaspá.

    Hlutir á ferð (bílar og plötur) halda áfram á þekktum hraða án þess að kassar
    stökkvi aftur á bak þegar ný greining berst, og án þess að gamlir kassar sitji
    eftir sem draugar (ghost boxes).
    Plötur geta tengst hreyfingu ökutækisins (anchors) svo þær fylgi bílnum
    hnökralaust jafnvel þótt ljósflæði á litlum skiltum missi punkta."""

    SMALL_W = 640
    HISTORY = 90

    def __init__(self, shrink: int = 8):
        self.shrink = shrink  # 8 = sleppa ytri hluta; 0 = nota allan kassann (fyrir plötur)
        self.hist = deque(maxlen=self.HISTORY)
        self.tracks = []
        self.det_id = None
        self.zoom = None
        self.scale = 1.0

    def gray(self, frame):
        h, w = frame.shape[:2]
        self.scale = min(1.0, self.SMALL_W / w)
        small = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _get_pts(self, g, box):
        """Finnur heppilega punkta til eftirbreytni innan kassans."""
        x1, y1, x2, y2 = (box * self.scale).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(g.shape[1], x2), min(g.shape[0], y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        mask = np.zeros_like(g)
        # Sleppa neðstu 15% (malbik undir dekkjum) svo punktar festist ekki á veginum
        my_top = max(1, int((y2 - y1) * 0.05)) if self.shrink else 0
        my_bot = max(2, int((y2 - y1) * 0.15)) if self.shrink else 0
        mx = max(1, int((x2 - x1) * 0.05)) if self.shrink else 0
        mask[max(0, y1 + my_top):max(0, y2 - my_bot), max(0, x1 + mx):max(0, x2 - mx)] = 255
        min_dist = 2 if self.shrink == 0 else 3
        return cv2.goodFeaturesToTrack(g, 40, 0.01, min_dist, mask=mask)

    def _flow(self, g0, g1, pts, vx=0.0, vy=0.0):
        if g0.shape != g1.shape or len(pts) == 0:
            return pts, np.zeros(len(pts), dtype=bool)
        use_init = abs(vx) > 1.0 or abs(vy) > 1.0
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW if use_init else 0
        guess = pts + np.array([[[vx * self.scale, vy * self.scale]]], dtype=np.float32) if use_init else None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, guess, winSize=(31, 31), maxLevel=4, flags=flags)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, nxt, pts.copy() if use_init else None, winSize=(31, 31), maxLevel=4, flags=flags)
        fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
        return nxt, (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 2.5)

    def _get_displacement(self, disps, prev_v=(0.0, 0.0)):
        """Aðskilur punkta á ökutæki á ferð frá kyrrstæðum malbiks- og bakgrunnspunktum."""
        if len(disps) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        mags = np.linalg.norm(disps, axis=1)
        max_mag = np.percentile(mags, 90)
        # Ef allir punktar hreyfast örlítið (< 1.5 px í 360p), er hluturinn kyrr
        if max_mag < 1.5:
            return np.median(disps, axis=0)
        # Hlutur á ferð: skilja punkta á bílnum frá kyrrstæðum malbikspunktum
        moving = mags >= max(2.0, 0.35 * max_mag)
        if np.sum(moving) >= 3:
            return np.median(disps[moving], axis=0)
        return np.median(disps, axis=0)

    def _catch_up(self, det_frame_no, g_now, boxes):
        """Greining tekur ~0,5 s, svo kassarnir í henni sýna hvar hlutirnir VORU. Hér eru punktar í hverjum
        kassa raktir gegnum geymdu rammana, frá rammanum sem greiningin var gerð á og fram í núverandi
        ramma. Skilar hliðrun (dx, dy) fyrir hvern kassa, eða None ef ekki tókst að rekja hann."""
        frames = [gg for n, gg in self.hist if n >= det_frame_no]
        if not frames or self.hist[0][0] > det_frame_no or frames[0].shape != g_now.shape:
            return [None] * len(boxes)
        chain = frames[::2]  # annar hver rammi dugar og helmingar vinnuna
        chain.append(g_now)
        pts = [self._get_pts(chain[0], b) for b in boxes]
        total = [np.zeros(2, np.float32) if p is not None and len(p) >= 4 else None for p in pts]
        last = [np.zeros(2, np.float32) for _ in boxes]
        done = [t is None for t in total]  # True = hætt að rekja (kyrrstæður hlutur eða punktar týndir)
        for step, (a, b) in enumerate(zip(chain, chain[1:])):
            live = [i for i in range(len(boxes)) if not done[i]]
            if not live:
                break
            allp = np.concatenate([pts[i] for i in live])
            guess = np.concatenate([pts[i] + (last[i] * self.scale).reshape(1, 1, 2) for i in live]).astype(np.float32)
            nxt, ok = self._flow_pts(a, b, allp, guess)
            k = 0
            for i in live:
                n = len(pts[i])
                p0, p1, good = pts[i], nxt[k:k + n], ok[k:k + n]
                k += n
                if good.sum() < 4:
                    total[i], done[i] = (total[i] if np.abs(total[i]).max() >= 1.0 and step >= 2 else None), True
                    continue
                disps = (p1[good] - p0[good]).reshape(-1, 2) / self.scale
                mags = np.linalg.norm(disps, axis=1)
                top = np.percentile(mags, 90)
                if top < 0.75:  # ekkert hreyfist í kassanum
                    if step >= 1 and np.abs(total[i]).max() < 1.0:
                        done[i] = True  # kyrrstæður hlutur: óþarfi að rekja lengra
                    pts[i] = p1[good]
                    continue
                moving = mags >= max(1.0, 0.35 * top)  # punktar á bílnum, ekki á götunni bak við hann
                if moving.sum() < 3:
                    pts[i] = p1[good]
                    continue
                d = np.median(disps[moving], axis=0)
                total[i] += d
                last[i] = d
                pts[i] = p1[good][moving]  # sleppa bakgrunnspunktum svo þeir trufli ekki næstu skref
                if len(pts[i]) < 4:
                    done[i] = True
        return total

    def _flow_pts(self, g0, g1, pts, guess):
        """Ljósflæði fyrir marga punkta í einu, með upphafságiskun fyrir hvern punkt."""
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, guess.copy(), winSize=(21, 21), maxLevel=3,
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, nxt, pts.copy(), winSize=(21, 21), maxLevel=3,
                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
        return nxt, (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 2.0)

    def _find_anchor(self, box, anchors):
        """Finnur ökutæki sem umlykur eða skarast við tiltekinn kassa (t.d. plötu)."""
        if not anchors:
            return None
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for a in anchors:
            b = a.bounding_box
            pad_x = (b.x2 - b.x1) * 0.15
            pad_y = (b.y2 - b.y1) * 0.15
            if (b.x1 - pad_x <= cx <= b.x2 + pad_x) and (b.y1 - pad_y <= cy <= b.y2 + pad_y):
                return a
        return None

    def _anchor_coords(self, box, a):
        """Reiknar hlutfallslega staðsetningu kassa innan ökutækis."""
        b = a.bounding_box
        aw = max(1.0, float(b.x2 - b.x1))
        ah = max(1.0, float(b.y2 - b.y1))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = float(box[2] - box[0]), float(box[3] - box[1])
        return {
            "tid": getattr(a, "tid", None),
            "rel_cx": (cx - b.x1) / aw,
            "rel_cy": (cy - b.y1) / ah,
            "rel_w": w / aw,
            "rel_h": h / ah,
        }

    def _apply_anchor(self, a, anc):
        """Færir kassa miðað við nýjustu staðsetningu ökutækisins."""
        b = a.bounding_box
        aw = float(b.x2 - b.x1)
        ah = float(b.y2 - b.y1)
        cx = b.x1 + anc["rel_cx"] * aw
        cy = b.y1 + anc["rel_cy"] * ah
        w = anc["rel_w"] * aw
        h = anc["rel_h"] * ah
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    def _step(self, g0, g1, anchors=None):
        if g0.shape != g1.shape:
            return

        anchor_by_tid = {getattr(a, "tid", None): a for a in anchors if getattr(a, "tid", None) is not None} if anchors else {}
        for t in self.tracks:
            anc = t.get("anchor")
            if anc and anc.get("tid") in anchor_by_tid:
                a = anchor_by_tid[anc["tid"]]
                t["box"] = self._apply_anchor(a, anc)
                t["vx"] = getattr(a, "vx", 0.0)
                t["vy"] = getattr(a, "vy", 0.0)
                t["moving"] = getattr(a, "moving", False)
            elif anchors and not anc:
                a = self._find_anchor(t["box"], anchors)
                if a is not None:
                    t["anchor"] = self._anchor_coords(t["box"], a)

        # Endurvekja punkta ef þeir eru orðnir of fáir
        for t in self.tracks:
            if t["pts"] is None or len(t["pts"]) < 6:
                t["pts"] = self._get_pts(g0, t["box"])

        for t in self.tracks:
            if t.get("anchor"):
                continue

            if t["pts"] is None or len(t["pts"]) < 4:
                vx, vy = t.get("vx", 0.0), t.get("vy", 0.0)
                t["box"] += (vx, vy, vx, vy)
                t["vx"] = vx * 0.98
                t["vy"] = vy * 0.98
                continue

            nxt, ok = self._flow(g0, g1, t["pts"], t.get("vx", 0.0), t.get("vy", 0.0))
            if ok.sum() < 4:
                vx, vy = t.get("vx", 0.0), t.get("vy", 0.0)
                t["box"] += (vx, vy, vx, vy)
                t["pts"] = None
                continue

            disps = (nxt[ok] - t["pts"][ok]).reshape(-1, 2) / self.scale
            d = self._get_displacement(disps, (t.get("vx", 0.0), t.get("vy", 0.0)))
            t["box"] += (d[0], d[1], d[0], d[1])
            t["pts"] = nxt[ok]
            t["vx"] = 0.5 * t.get("vx", d[0]) + 0.5 * d[0]
            t["vy"] = 0.5 * t.get("vy", d[1]) + 0.5 * d[1]
            if abs(t["vx"]) > 0.3:
                t["moving"] = True

    def update(self, frame_no, frame, det, zoom, anchors=None):
        """det = (rammanúmer, aðdráttur, hlutir) frá greiningarþræðinum eða None."""
        g = self.gray(frame)
        if zoom != self.zoom:
            self.hist.clear()
            self.tracks, self.zoom = [], zoom

        if det is not None and det[0] != self.det_id and det[1] == zoom:
            det_frame_no = det[0]
            self.det_id = det_frame_no
            df = max(0, frame_no - det_frame_no)

            # Ef eldri rammi er til, keyra fyrst ljósflæði fram að núverandi ramma
            if self.hist:
                self._step(self.hist[-1][1], g, anchors=anchors)

            det_objs = det[2]
            shifts = self._catch_up(det_frame_no, g, [
                np.array([o.bounding_box.x1, o.bounding_box.y1, o.bounding_box.x2, o.bounding_box.y2], np.float32)
                for o in det_objs])
            new_tracks = []
            used_track_indices = set()
            used_det_indices = set()

            # Umferð 1: Tengja nákvæmlega eftir tid (fyrir ökutæki sem VehicleTracks þekkir)
            for di, o in enumerate(det_objs):
                o_tid = getattr(o, "tid", None)
                if o_tid is not None:
                    for ti, t in enumerate(self.tracks):
                        if ti not in used_track_indices and t.get("tid") == o_tid:
                            used_track_indices.add(ti)
                            used_det_indices.add(di)
                            b = o.bounding_box
                            raw_box = np.array([b.x1, b.y1, b.x2, b.y2], np.float32)

                            vx = getattr(o, "vx", 0.0) / 30.0 if hasattr(o, "vx") and abs(o.vx) > 0.01 else t.get("vx", 0.0)
                            vy = getattr(o, "vy", 0.0) / 30.0 if hasattr(o, "vy") and abs(o.vy) > 0.01 else t.get("vy", 0.0)
                            proj_box = raw_box + np.array([vx * df, vy * df, vx * df, vy * df], np.float32)

                            # Ef greining kemur seint (df > 0), nota spá fram til núna í stað fortíðarhnita
                            sh = shifts[di]
                            if sh is not None:  # greiningin rakin fram til núna: nota hana beint
                                t["box"] = raw_box + np.array([sh[0], sh[1], sh[0], sh[1]], np.float32)
                                if df > 0:
                                    vx, vy = float(sh[0]) / df, float(sh[1]) / df
                            elif df > 0 and (abs(vx) > 0.1 or abs(vy) > 0.1):
                                t["box"] = 0.5 * t["box"] + 0.5 * proj_box
                            else:
                                t["box"] = 0.2 * t["box"] + 0.8 * raw_box

                            t["label"] = o.label
                            t["conf"] = o.confidence
                            t["color"] = getattr(o, "color", t.get("color"))
                            t["moving"] = getattr(o, "moving", t.get("moving", False))
                            if sh is not None and df > 0:
                                t["vx"], t["vy"] = vx, vy
                            else:
                                t["vx"] = vx if abs(vx) > 0.01 else t.get("vx", 0.0)
                                t["vy"] = vy if abs(vy) > 0.01 else t.get("vy", 0.0)
                            t["age"] = 0
                            t["pts"] = self._get_pts(g, t["box"])
                            new_tracks.append(t)
                            break

            # Umferð 2: Tengja óparaðar greiningar eftir rúmfræðilegri nálægð / skörun
            for di, o in enumerate(det_objs):
                if di in used_det_indices:
                    continue
                b = o.bounding_box
                raw_box = np.array([b.x1, b.y1, b.x2, b.y2], np.float32)
                sh = shifts[di]
                if sh is not None:  # hér eftir er raw_box staðsetningin NÚNA, ekki þegar greiningin var gerð
                    raw_box = raw_box + np.array([sh[0], sh[1], sh[0], sh[1]], np.float32)
                best_ti, best_score = None, 1e9

                o_cx = (raw_box[0] + raw_box[2]) / 2
                o_cy = (raw_box[1] + raw_box[3]) / 2
                o_w = max(1.0, raw_box[2] - raw_box[0])
                o_h = max(1.0, raw_box[3] - raw_box[1])

                for ti, t in enumerate(self.tracks):
                    if ti in used_track_indices:
                        continue
                    if (t["label"] == "person") != (o.label == "person"):
                        continue

                    tb = t["box"]
                    t_cx = (tb[0] + tb[2]) / 2
                    t_cy = (tb[1] + tb[3]) / 2
                    dist_norm = hypot((o_cx - t_cx) / o_w, (o_cy - t_cy) / o_h)

                    ix = max(0, min(raw_box[2], tb[2]) - max(raw_box[0], tb[0]))
                    iy = max(0, min(raw_box[3], tb[3]) - max(raw_box[1], tb[1]))
                    has_inter = ix > 0 and iy > 0

                    if dist_norm < 1.2 or has_inter:
                        score = dist_norm - (0.5 if has_inter else 0.0)
                        if score < best_score:
                            best_score, best_ti = score, ti

                if best_ti is not None:
                    used_track_indices.add(best_ti)
                    used_det_indices.add(di)
                    t = self.tracks[best_ti]
                    vx = t.get("vx", 0.0)
                    vy = t.get("vy", 0.0)
                    proj_box = raw_box + np.array([vx * df, vy * df, vx * df, vy * df], np.float32)
                    if sh is not None:
                        t["box"] = raw_box
                        if df > 0:
                            t["vx"], t["vy"] = float(sh[0]) / df, float(sh[1]) / df
                    elif df > 0 and (abs(vx) > 0.1 or abs(vy) > 0.1):
                        t["box"] = 0.5 * t["box"] + 0.5 * proj_box
                    else:
                        t["box"] = 0.2 * t["box"] + 0.8 * raw_box

                    t["label"] = o.label
                    t["conf"] = o.confidence
                    t["tid"] = getattr(o, "tid", t.get("tid"))
                    t["color"] = getattr(o, "color", t.get("color"))
                    t["moving"] = getattr(o, "moving", t.get("moving", False))
                    t["age"] = 0
                    t["pts"] = self._get_pts(g, t["box"])
                    new_tracks.append(t)
                else:
                    # Ný greining
                    vx = getattr(o, "vx", 0.0) / 30.0 if hasattr(o, "vx") else 0.0
                    vy = getattr(o, "vy", 0.0) / 30.0 if hasattr(o, "vy") else 0.0
                    if sh is not None:
                        box = raw_box
                        if df > 0:
                            vx, vy = float(sh[0]) / df, float(sh[1]) / df
                    else:
                        box = raw_box + np.array([vx * df, vy * df, vx * df, vy * df], np.float32)
                    pts = self._get_pts(g, box)
                    a = self._find_anchor(box, anchors) if anchors else None
                    anc = self._anchor_coords(box, a) if a is not None else None
                    new_tracks.append({
                        "label": o.label, "conf": o.confidence, "pts": pts,
                        "tid": getattr(o, "tid", None), "color": getattr(o, "color", None),
                        "moving": getattr(o, "moving", getattr(a, "moving", False) if a else False),
                        "box": box, "anchor": anc,
                        "vx": vx, "vy": vy, "age": 0
                    })

            # Varðveita óparaða kassa:
            # Ökutæki sem þegar eru skráð í new_tracks mega ALDREI koma aftur sem draugakassar!
            claimed_tids = {t["tid"] for t in new_tracks if t.get("tid") is not None}
            anchor_by_tid = {getattr(a, "tid", None): a for a in anchors if getattr(a, "tid", None) is not None} if anchors else {}
            for ti, t in enumerate(self.tracks):
                if ti in used_track_indices:
                    continue
                if t.get("tid") in claimed_tids:
                    continue  # Koma í veg fyrir tvítalningu eða draugakassa

                anc = t.get("anchor")
                if anc and anc.get("tid") in anchor_by_tid:
                    a = anchor_by_tid[anc["tid"]]
                    t["box"] = self._apply_anchor(a, anc)
                    t["vx"] = getattr(a, "vx", 0.0)
                    t["vy"] = getattr(a, "vy", 0.0)
                    t["moving"] = getattr(a, "moving", False)
                    t["age"] = 0
                    new_tracks.append(t)
                    continue

                max_age = 8 if t.get("moving", False) else 25
                if t.get("age", 0) < max_age:
                    t["age"] = t.get("age", 0) + 1
                    vx, vy = t.get("vx", 0.0), t.get("vy", 0.0)
                    t["box"] += (vx, vy, vx, vy)
                    new_tracks.append(t)

            self.tracks = new_tracks
        elif self.hist:
            self._step(self.hist[-1][1], g, anchors=anchors)
            for t in self.tracks:
                t["age"] = t.get("age", 0) + 1

        w, h = frame.shape[1], frame.shape[0]
        # Hreinsa kassa sem eru komnir út fyrir mynd eða orðnir of gamlir
        self.tracks = [
            t for t in self.tracks
            if (t.get("age", 0) < (15 if t.get("moving", False) else 45))
            and t["box"][2] > -20 and t["box"][0] < w + 20
            and t["box"][3] > -20 and t["box"][1] < h + 20
        ]
        self.hist.append((frame_no, g))
        return [
            SimpleNamespace(
                label=t["label"], confidence=t["conf"], tid=t.get("tid"),
                moving=t.get("moving", False), color=t.get("color"),
                vx=t.get("vx", 0.0), vy=t.get("vy", 0.0),
                bounding_box=SimpleNamespace(x1=t["box"][0], y1=t["box"][1],
                                             x2=t["box"][2], y2=t["box"][3])
            )
            for t in self.tracks
        ]
