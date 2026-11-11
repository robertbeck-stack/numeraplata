"""Litagreining og útlitsfingraför fyrir ökutæki með hvítjöfnun miðað við malbik."""
import cv2
import numpy as np

COLOR_MIN_CHROMA = 10  # minnsti litstyrkur (Lab) umfram slikju birtunnar svo díll teljist litaður
COLOR_MIN_REL = 0.15   # ...og minnsti litstyrkur sem hlutfall af birtu dílsins (ljósir fletir þurfa meiri lit)
COLOR_BLACK, COLOR_WHITE = 70, 130  # birtumörk (0-255): dekkra = svartur, ljósara = hvítur, á milli = grár
APPEAR_WEIGHT = 3.0   # vægi útlits (litar) á móti fjarlægð þegar kassi er paraður við bíl
APPEAR_MIN = 0.5      # minni líkindi en þetta: ekki sami bíllinn, sama hversu nálægt kassinn er
APPEAR_PARKED_MIN = 0.8  # sama, fyrir bíl sem hefur staðið kyrr lengi

COLOR_ANGLES = (
    (75, "rauður"),
    (90, "appelsínugulur"),
    (115, "gulur"),
    (200, "grænn"),
    (320, "blár"),
    (345, "fjólublár"),
    (361, "rauður"),
)  # efri mörk litahorns í Lab (gráður) og heiti


def appearance(frame, b):
    """Litafingrafar bíls: gróft litasúlurit (litblær x mettun x birta) úr miðhluta kassans.
    Hvítur jeppi og dökkur sendibíll fá mjög ólík fingraför, sami bíll svipað milli ramma."""
    fh, fw = frame.shape[:2]
    w, h = b.x2 - b.x1, b.y2 - b.y1
    x1, x2 = int(max(0, b.x1 + 0.1 * w)), int(min(fw, b.x2 - 0.1 * w))
    y1, y2 = int(max(0, b.y1 + 0.2 * h)), int(min(fh, b.y2 - 0.1 * h))
    if x2 - x1 < 6 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (24, 12), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [6, 3, 5], [0, 180, 0, 256, 0, 256]).ravel()
    return np.sqrt(hist / max(1.0, hist.sum()))  # rót: þá er innfeldi tveggja fingrafara líkindin (0-1)


def scene_cast(frame):
    """Litaslikja og birtustig (a, b, L): miðgildi í neðri helmingi myndar. Notað sem varaviðmið þegar
    ekki sést í malbik undir bílnum, og til að meta hvort bíll er ljós eða dökkur miðað við birtu dagsins."""
    h = frame.shape[0]
    lab = cv2.cvtColor(cv2.resize(frame[h // 2:], (64, 18), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
    return float(np.median(lab[..., 1])), float(np.median(lab[..., 2])), float(np.median(lab[..., 0]))


def color_stats(frame, b, cast=(128.0, 128.0, 100.0)):
    """Skilar (hlutfalli litaðra díla, litaheiti þeirra, birtu bílsins, birtu miðað við malbikið) eða None.

    Viðmiðið fyrir 'litlaust' er malbikið rétt undir bílnum: það sýnir lit birtunnar á þeim stað
    (blá í rökkri og dögun, gul undir ljósastaur). Slikjan vex með birtu flatarins, svo hún er sköluð
    með birtu hvers díls áður en hún er dregin frá. Annars yrði hvítur bíll í dögun 'blár'."""
    fh, fw = frame.shape[:2]
    w, h = b.x2 - b.x1, b.y2 - b.y1
    x1, x2 = int(max(0, b.x1 + 0.15 * w)), int(min(fw, b.x2 - 0.15 * w))
    y1, y2 = int(max(0, b.y1 + 0.35 * h)), int(min(fh, b.y2 - 0.25 * h))
    if x2 - x1 < 6 or y2 - y1 < 4:
        return None
    lab = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (32, 16), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
    light = lab[..., 0].ravel().astype(float)
    refs = [cast]
    uy1, uy2 = int(b.y2 + 0.03 * h), int(min(fh, b.y2 + 0.25 * h))
    ux1, ux2 = int(max(0, b.x1)), int(min(fw, b.x2))
    if uy2 - uy1 >= 3 and ux2 - ux1 >= 6:  # malbikið undir bílnum
        under = cv2.cvtColor(cv2.resize(frame[uy1:uy2, ux1:ux2], (16, 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
        refs.append((float(np.median(under[..., 1])), float(np.median(under[..., 2])), float(np.median(under[..., 0]))))
    best = None
    for ref_a, ref_b, ref_l in refs:
        k = np.clip(light / max(ref_l, 1.0), 0.3, 3.0)
        ca = (lab[..., 1].ravel() - 128.0) - (ref_a - 128.0) * k
        cb = (lab[..., 2].ravel() - 128.0) - (ref_b - 128.0) * k
        chroma = np.hypot(ca, cb)
        vivid = (chroma > COLOR_MIN_CHROMA) & (chroma > COLOR_MIN_REL * light) & (light > 20)
        # Flestir bílar eru hvítir, gráir eða svartir. Bíll telst því aðeins litaður ef hann er það miðað við
        # bæði viðmiðin (malbikið undir honum og myndina í heild): það viðmið gildir sem gefur minni lit.
        if best is None or vivid.mean() < best[0]:
            best = (float(vivid.mean()), ca[vivid], cb[vivid], ref_l)
    share, ca, cb, ref_l = best
    name = None
    if len(ca):
        ang = np.degrees(np.arctan2(cb, ca)) % 360
        votes = {}
        for i in np.searchsorted([u for u, _ in COLOR_ANGLES], ang, side="right"):
            votes[COLOR_ANGLES[i][1]] = votes.get(COLOR_ANGLES[i][1], 0) + 1
        name = max(votes, key=votes.get)
    v = float(np.median(light))
    return share, name, v * (100.0 / max(25.0, cast[2])) ** 0.5, v / max(refs[-1][2], 1.0)


def color_name(frame, b, cast=(128.0, 128.0, 100.0)):
    """Litaflokkur bíls: hvítur / grár / svartur, eða litur ef nógu stór hluti bílsins er litaður.
    Miðað er við miðhluta kassans (yfirbyggingin), ekki rúður efst eða götu og dekk neðst."""
    st = color_stats(frame, b, cast)
    if st is None:
        return None
    share, name, v, rel = st
    # hvítur bíll í sól er gulleitur miðað við skuggann undir honum: gult þarf því að vera mun skýrara
    if share >= (0.6 if name in ("gulur", "appelsínugulur") else 0.3):
        return name
    if v >= COLOR_WHITE or (v >= 0.8 * COLOR_WHITE and rel >= 1.7):  # ljós, eða mun ljósari en malbikið undir
        return "hvítur"
    return "svartur" if v < COLOR_BLACK else "grár"


def similar(a, b) -> float:
    """Reiknar líkindi tveggja útlitsfingrafara (0-1)."""
    return 1.0 if a is None or b is None else float((a * b).sum())
