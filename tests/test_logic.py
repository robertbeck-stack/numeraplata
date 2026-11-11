"""Prófanir á rökfræði Numby ALPR sem þurfa hvorki myndavél né líkön.
Keyrsla:  python tests/test_logic.py"""
import os
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import alpr

os.chdir(tempfile.mkdtemp())  # prófin skrifa csv-skrár og myndir: ekki í verkefnismöppuna


def box(x1, y1, x2, y2):
    return NS(x1=x1, y1=y1, x2=x2, y2=y2)


def test_valid_plate():
    assert alpr.valid_plate("AB-123") == "AB123"
    assert alpr.valid_plate("IS AB 123") == "AB123"   # landsmerkið klippt af
    assert alpr.valid_plate("IS123") == "IS123"       # ...en ekki þegar það er hluti af númerinu
    assert alpr.valid_plate("AB 12O") == "AB120"
    assert alpr.valid_plate("PARIS") is None


def test_logging_rules():
    """Þrír eins lestrar þarf; kyrrstæður bíll skráist einu sinni; sama plata tvisvar í ramma = eitt atkvæði."""
    alpr.VEHICLE_CHECK = False
    clock = [1000.0]
    real_time = alpr.time
    alpr.time = NS(time=lambda: clock[0])
    try:
        img, logs = np.zeros((1080, 1920, 3), np.uint8), []

        def see(text, conf=0.9, n=1):
            r = NS(ocr=NS(text=text, confidence=conf), detection=NS(bounding_box=box(100, 100, 220, 140)))
            alpr.process_frame(None, img, logs, [r] * n)

        see("XY987", n=2)
        clock[0] += 0.1
        see("XY987", n=2)
        assert len(logs) == 0, "tveir rammar eiga ekki að duga"
        clock[0] += 0.1
        see("XY987")
        assert [e["plate"] for e in logs] == ["XY987"]
        for _ in range(600):  # kyrrstæður í 5 mínútur
            clock[0] += 0.5
            see("XY987")
        assert len(logs) == 1, "kyrrstæður bíll á ekki að skrást aftur"
        clock[0] += alpr.RESIGHT_SECONDS + 1
        for _ in range(3):
            clock[0] += 0.5
            see("XY987")
        assert len(logs) == 2, "bíll sem kemur aftur seinna á að skrást"
        logs.clear()
        alpr._pending.clear()
        for _ in range(6):  # stöðugur lestur en lágt öryggi: ekki skráð
            clock[0] += 0.1
            see("ZZ111", conf=0.55)
        assert logs == []
    finally:
        alpr.time = real_time


def test_people_counter():
    c = alpr.PeopleCounter()
    for n in (0, 2, 3, 1):
        c.update(n)
    assert c.now == 1 and c.peak == 3


def simulate(cars, occluders, seconds=14.0, step=0.5, frame_w=1920):
    """cars: (x við t=0, hraði px/s, breidd). Bíll sem er bak við fyrirstöðu sést aðeins að hluta eða ekki."""
    if os.path.exists(alpr.TRAFFIC_FILE):
        os.remove(alpr.TRAFFIC_FILE)  # annars byrjar teljarinn á talningu fyrri prófunar
    tr = alpr.VehicleTracks()
    ids = {i: set() for i in range(len(cars))}
    t = 0.0
    while t <= seconds:
        objs = [NS(label="truck", confidence=0.95, bounding_box=box(a, 500, b, 800)) for a, b in occluders]
        for i, (x0, v, w) in enumerate(cars):
            vis = [(max(0.0, x0 + v * t), min(float(frame_w), x0 + v * t + w))]
            for a, b in occluders:  # klippa burt það sem fyrirstaðan hylur
                vis = [p for s, e in vis for p in ((s, min(e, a)), (max(s, b), e)) if p[1] - p[0] > 25]
            for s, e in vis[:1]:
                o = NS(label="car", confidence=0.9, bounding_box=box(s, 600, e, 760))
                o.car = i
                objs.append(o)
        tr.update(objs, t, frame_w)
        for o in objs:
            if hasattr(o, "car") and getattr(o, "tid", None) and (o.bounding_box.x2 - o.bounding_box.x1) > 0.8 * cars[o.car][2]:
                ids[o.car].add(o.tid)
        t += step
    return tr, ids


def test_vehicle_tracks():
    line = alpr.COUNT_LINE * 1920
    # bíll keyrir til hægri, alveg falinn bak við kyrrstæðan vörubíl á leiðinni
    tr, ids = simulate([(-300, 220, 300)], [(700, 1150)])
    assert (tr.left, tr.right) == (0, 1), (tr.left, tr.right)
    assert len(ids[0]) == 1, f"bíllinn á að halda sama númeri gegnum feluna, fékk {ids[0]}"
    # tveir bílar mætast bak við vörubílinn
    tr, ids = simulate([(-300, 220, 300), (1920, -220, 300)], [(700, 1150)])
    assert (tr.left, tr.right) == (1, 1), (tr.left, tr.right)
    # kyrrstæðir bílar teljast aldrei
    tr, _ = simulate([(line - 150, 0, 300), (300, 0, 300)], [])
    assert (tr.left, tr.right) == (0, 0)


def test_night_counter():
    counted = []
    nc = alpr.NightCounter(lambda label, direction: counted.append(direction))
    dark = np.full((720, 1280, 3), 12, np.uint8)
    dark[600:] = 150  # upplýstur forgrunnur, eins og stéttin fyrir framan gluggann
    fps, t = 30.0, 0.0
    for i in range(int(8 * fps)):
        f = dark.copy()
        if i > 3 * fps:  # bíll til vinstri, 600 px/s: hvít framljós og rauð afturljós 150 px á eftir
            x = int(1500 - 600 * (i / fps - 3))
            if 0 <= x < 1250:
                f[440:446, x:x + 30] = 255
            if 0 <= x + 150 < 1250:
                f[441:445, x + 150:x + 175] = (0, 0, 230)
        nc.update(f, t)
        t += 1 / fps
    assert nc.night
    assert counted == ["vinstri"], counted  # fram- og afturljós = einn bíll
    day = alpr.NightCounter(lambda *a: counted.append("x"))
    bright = np.full((720, 1280, 3), 200, np.uint8)
    for i in range(90):
        day.update(bright, i / 30)
    assert not day.night


def test_is_same_car_distinction():
    """Athugar að OCR-stafavíxl séu talin sami bíll, en ólíkir bílar í röð ekki."""
    assert alpr.is_same_car("VSD07", "VSO07")    # D vs O
    assert alpr.is_same_car("XLS53", "IXL53")    # I-fánamerki fremst
    assert alpr.is_same_car("AB120", "AB12O")    # 0 vs O
    assert alpr.is_same_car("AB128", "AB12B")    # 8 vs B
    assert alpr.is_same_car("AB125", "AB12S")    # 5 vs S
    # Ólíkir bílar sem eiga EKKI að teljast sami bíll
    assert not alpr.is_same_car("WA936", "WA937")
    assert not alpr.is_same_car("AB123", "AC123")
    assert not alpr.is_same_car("VSD07", "ABC12")


def test_three_letter_sticker_plate():
    """Styður bæði 2-stafa og 3-stafa plötur með skoðunarmiða."""
    assert alpr.valid_plate("ABC 24 12") == "ABC12"
    assert alpr.valid_plate("AB 24 123") == "AB123"


def test_safe_path():
    """Koma í veg fyrir path traversal árásum."""
    assert alpr.is_safe_path("captures", "mynd.jpg")
    assert not alpr.is_safe_path("captures", "../plates.csv")
    assert not alpr.is_safe_path("captures", "..\\plates.csv")
    assert not alpr.is_safe_path("captures", "/captures/../../secret.txt")


def test_box_tracker_motion():
    """Athugar að BoxTracker elti bíl á mikilli ferð í stað þess að frjósa."""
    import cv2
    w, h = 1920, 1080
    frames = []
    for i in range(10):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        cx = int(200 + i * 30)
        cv2.rectangle(img, (cx - 150, 440), (cx + 150, 560), (0, 0, 180), -1)
        cv2.circle(img, (cx - 90, 560), 25, (10, 10, 10), -1)
        cv2.circle(img, (cx + 90, 560), 25, (10, 10, 10), -1)
        cv2.circle(img, (cx - 90, 560), 12, (200, 200, 200), -1)
        cv2.circle(img, (cx + 90, 560), 12, (200, 200, 200), -1)
        frames.append(img)

    tracker = alpr.BoxTracker()
    b0 = box(50, 440, 350, 560)
    tracker.update(0, frames[0], (0, 1.0, [NS(label="car", confidence=0.9, tid=1, bounding_box=b0)]), 1.0)
    for f in range(1, 10):
        res = tracker.update(f, frames[f], None, 1.0)
        tracked_cx = (res[0].bounding_box.x1 + res[0].bounding_box.x2) / 2
        actual_cx = 200 + f * 30
        assert abs(tracked_cx - actual_cx) < 5.0, f"Ramminn dróst aftur úr: {tracked_cx} vs {actual_cx}"


def test_box_tracker_tid_match_no_ghosts():
    """Athugar að ný greining á sama bíl (sama tid) búi ekki til tvöfalda draugakassa."""
    tracker = alpr.BoxTracker()
    f0 = np.full((720, 1280, 3), 50, dtype=np.uint8)
    det0 = (0, 1.0, [NS(label="car", confidence=0.9, tid=5, moving=True, vx=600.0, vy=0.0,
                        bounding_box=box(100, 200, 250, 350))])
    tracker.update(0, f0, det0, 1.0)

    # Bíll tekur stórt stökk (t.d. eftir hálfa sekúndu af greiningu án skörunar)
    f10 = np.full((720, 1280, 3), 50, dtype=np.uint8)
    det1 = (10, 1.0, [NS(label="car", confidence=0.9, tid=5, moving=True, vx=600.0, vy=0.0,
                         bounding_box=box(600, 200, 750, 350))])
    res = tracker.update(10, f10, det1, 1.0)
    assert len(res) == 1, f"Átti aðeins að vera 1 kassi en fékk {len(res)}"
    assert res[0].tid == 5
    assert res[0].bounding_box.x1 >= 500


def test_plate_tracker_anchored_motion():
    """Athugar að plötukassi fylgi bíl á ferð jafnvel þótt ALPR lesi ekki plötuna á öllum römmum."""
    pt = alpr.BoxTracker(shrink=0)
    f0 = np.full((720, 1280, 3), 50, dtype=np.uint8)
    det0 = (0, 1.0, [NS(label="AB123", confidence=0.95, bounding_box=box(150, 280, 200, 300))])
    veh0 = [NS(tid=1, moving=True, vx=20.0, vy=0.0, bounding_box=box(100, 200, 250, 350))]
    pt.update(0, f0, det0, 1.0, anchors=veh0)

    # 10 rammar líða, bíllinn færst um 200 px, ALPR greinir enga plötu á millirömmum
    f10 = np.full((720, 1280, 3), 50, dtype=np.uint8)
    veh10 = [NS(tid=1, moving=True, vx=20.0, vy=0.0, bounding_box=box(300, 200, 450, 350))]
    res = pt.update(10, f10, (10, 1.0, []), 1.0, anchors=veh10)
    assert len(res) == 1, "Platan átti að haldast virk"
    plate_cx = (res[0].bounding_box.x1 + res[0].bounding_box.x2) / 2
    assert plate_cx > 300, f"Platan átti að fylgja bílnum áfram, en var við {plate_cx}"


def test_fast_bus_with_occluding_parked_car():
    """Athugar að stór rúta á mikilli ferð sem keyrir framhjá kyrrstæðum bíl haldi sínu númeri
    og ræni ekki númeri kyrrstæða bílsins."""
    tr = alpr.VehicleTracks()
    objs0 = [
        NS(label="bus", confidence=0.88, bounding_box=box(610, 550, 1048, 660)),
        NS(label="car", confidence=0.83, bounding_box=box(990, 550, 1120, 660)),
    ]
    tr.update(objs0, 0.0, 1920)
    bus_tid = objs0[0].tid
    car_tid = objs0[1].tid
    assert bus_tid != car_tid

    # 0.3s síðar: rútan hefur færst um 206 px (mjög hratt ~700 px/s) og er nú beint fyrir aftan kyrrstæða bílinn (dx=24 px)
    objs1 = [
        NS(label="bus", confidence=0.68, bounding_box=box(816, 550, 1254, 660)),
        NS(label="car", confidence=0.83, bounding_box=box(990, 550, 1120, 660)),
    ]
    tr.update(objs1, 0.3, 1920)
    assert objs1[0].tid == bus_tid, f"Rútan átti að halda tid={bus_tid}, en fékk {objs1[0].tid}"
    assert objs1[1].tid == car_tid, f"Kyrrstæði bíllinn átti að halda tid={car_tid}, en fékk {objs1[1].tid}"
    assert objs1[0].label == "bus"
    assert objs1[1].label == "car"


def test_motion_tracker():
    """Hreyfirakning: bíll á fullri ferð fær kassa sem fylgir honum og telst einu sinni yfir línuna."""
    from numby.motion import MotionTracker
    crossed = []
    mt = MotionTracker(0.75, lambda direction, mid: crossed.append(direction))
    rng = np.random.RandomState(1)
    bg = rng.randint(60, 90, (540, 960, 3)).astype(np.uint8)  # kyrr bakgrunnur með áferð
    worst = 0.0
    for i in range(120):
        f = bg.copy()
        x = int(-120 + 14 * i)  # 14 dílar á ramma til hægri
        if i >= 10 and x < 960:
            f[300:340, max(0, x):max(0, min(960, x + 110))] = 235
        objs = mt.update(f, i / 30.0, i)
        if i >= 10 and 60 < x < 800 and i > 25:
            assert objs, f"enginn kassi á bíl á ferð í ramma {i}"
            b = objs[0].bounding_box
            worst = max(worst, abs((b.x1 + b.x2) / 2 - (x + 55)))
            assert objs[0].vx > 5
    assert worst < 25, f"kassinn dregst {worst:.0f} díla aftur úr"
    assert crossed == ["hægri"], crossed
    # kyrr mynd: ekkert á ferð og ekkert talið
    still = MotionTracker(0.75, lambda direction, mid: crossed.append("x"))
    assert all(not still.update(bg, i / 30.0, i) for i in range(60))
    assert crossed == ["hægri"]


def test_count_dedupe():
    """Bílagreining og hreyfirakning telja sama bílinn: aðeins fyrri talningin gildir."""
    if os.path.exists(alpr.TRAFFIC_FILE):
        os.remove(alpr.TRAFFIC_FILE)
    tr = alpr.VehicleTracks()
    assert tr._count("motion", "hægri", "motion") is True
    assert tr._count("car", "hægri", "det") is False      # sami bíll, hinn teljarinn
    assert tr._count("motion", "vinstri", "motion") is True  # önnur átt er annar bíll
    assert tr._count("motion", "hægri", "motion") is True    # sami teljari aftur: nýr bíll
    assert (tr.left, tr.right) == (1, 2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
    print("allt í lagi")


