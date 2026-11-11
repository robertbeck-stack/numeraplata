"""Teikniaðgerðir fyrir OpenCV myndglugga og skyndimyndir."""
import cv2
import numpy as np

from .traffic import VEHICLE_NAMES

PANEL_W = 380
ROW_H = 84
THUMB_W, THUMB_H = 128, 72


def clean_text_for_opencv(text: str) -> str:
    """Umritar íslenska sérstafi í ASCII svo þeir brenglist ekki í cv2.putText (Hershey font)."""
    replacements = {
        "á": "a", "Á": "A",
        "ð": "d", "Ð": "D",
        "é": "e", "É": "E",
        "í": "i", "Í": "I",
        "ó": "o", "Ó": "O",
        "ú": "u", "Ú": "U",
        "ý": "y", "Ý": "Y",
        "þ": "th", "Þ": "Th",
        "æ": "ae", "Æ": "Ae",
        "ö": "o", "Ö": "O",
        "←": "<-", "→": "->",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def draw_vehicles(frame, vehicles) -> None:
    """Rauður kassi utan um hvern greindan bíl, með tegund, öryggi og stefnuör ef á hreyfingu."""
    for r in vehicles:
        b = r.bounding_box
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 230), 3)
        moving = getattr(r, "moving", False) or abs(getattr(r, "vx", 0.0)) > 0.2
        vx = getattr(r, "vx", 0.0)
        tid = getattr(r, "tid", None) if moving else None

        # Stefnumerking við hliðina á lit:
        # Ef bíll hreyfist: -> eða <-
        # Ef bíll stendur kyrr: (kyrr)
        if vx > 0.2:
            dir_sym = " ->"
        elif vx < -0.2:
            dir_sym = " <-"
        else:
            dir_sym = " (kyrr)"

        color = getattr(r, "color", None)
        color_part = f" {color}{dir_sym}" if color else dir_sym
        conf = f" {r.confidence * 100:.0f}%" if r.confidence is not None else ""  # hreyfirakning hefur ekkert öryggi
        label = f"{VEHICLE_NAMES.get(r.label, r.label)}{f' #{tid}' if tid else ''}{conf}{color_part}"
        ascii_label = clean_text_for_opencv(label)
        (tw, th), _ = cv2.getTextSize(ascii_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = y1 - 8 if y1 - th - 12 > 0 else y1 + th + 8
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 10, ty + 6), (40, 40, 230), -1)
        cv2.putText(frame, ascii_label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Teikna skýra stefnuör í miðju kassans ef bíllinn er á ferð
        if abs(vx) > 0.3 and (x2 - x1) > 40:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            arr_len = min(50, max(24, int((x2 - x1) * 0.25)))
            ax1 = cx - arr_len // 2 if vx > 0 else cx + arr_len // 2
            ax2 = cx + arr_len // 2 if vx > 0 else cx - arr_len // 2
            # Svartur skuggi svo örin sjáist skýrt á bæði ljósum og dökkum bílum
            cv2.arrowedLine(frame, (ax1, cy), (ax2, cy), (0, 0, 0), 4, tipLength=0.35)
            cv2.arrowedLine(frame, (ax1, cy), (ax2, cy), (0, 255, 255), 2, tipLength=0.35)


def draw_hidden(frame, hidden) -> None:
    """Strikaður kassi þar sem bíll sem sést ekki (t.d. bak við annan) ætti að vera samkvæmt spá."""
    h, w = frame.shape[:2]
    for r in hidden:
        b = r.bounding_box
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        if x2 < 0 or x1 > w:
            continue
        for x in range(x1, x2, 28):
            cv2.line(frame, (x, y1), (min(x + 14, x2), y1), (40, 40, 230), 2)
            cv2.line(frame, (x, y2), (min(x + 14, x2), y2), (40, 40, 230), 2)
        for y in range(y1, y2, 28):
            cv2.line(frame, (x1, y), (x1, min(y + 14, y2)), (40, 40, 230), 2)
            cv2.line(frame, (x2, y), (x2, min(y + 14, y2)), (40, 40, 230), 2)
        cv2.putText(frame, clean_text_for_opencv(f"#{r.tid} falinn"), (x1 + 4, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 230), 2)


def draw_people(frame, people) -> None:
    """Gulur kassi utan um fólk. Aðeins teiknað í lifandi mynd, ekkert vistað."""
    for r in people:
        b = r.bounding_box
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 210, 255), 2)


def draw_results(frame, results) -> None:
    """Teiknar ramma og lesinn texta á myndina út frá þegar reiknuðum niðurstöðum."""
    for res in results:
        box = getattr(res.detection, "bounding_box", None)
        if box is None:
            continue
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
        if not (res.ocr and res.ocr.text):
            continue
        c = res.ocr.confidence
        conf = float(np.mean(c)) if isinstance(c, (list, np.ndarray)) else float(c or 0)
        label = clean_text_for_opencv(f"{res.ocr.text} {conf * 100:.0f}%")
        y = y1 - 10 if y1 > 30 else y2 + 25
        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)


def build_panel(height: int, logs: list[dict], thumbs: dict, info: str = "") -> np.ndarray:
    """Hliðarspjald með nýjustu skráðu númerunum: smámynd, númer, tími og öryggi."""
    panel = np.full((height, PANEL_W, 3), 30, np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 44), (18, 18, 18), -1)
    cv2.putText(panel, f"Atvik ({len(logs)})", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (252, 134, 187), 2)
    y = 52
    if info:
        cv2.rectangle(panel, (0, 44), (PANEL_W, 74), (24, 24, 24), -1)
        cv2.putText(panel, clean_text_for_opencv(info), (14, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y = 82
    for e in logs:
        if y + ROW_H > height:
            break
        path = e["img"].lstrip("/\\")
        if path not in thumbs:
            img = cv2.imread(path)
            thumbs[path] = cv2.resize(img, (THUMB_W, THUMB_H)) if img is not None else None
        thumb = thumbs[path]
        if thumb is not None:
            panel[y + 6:y + 6 + THUMB_H, 10:10 + THUMB_W] = thumb
        tx = 10 + THUMB_W + 14
        cv2.putText(panel, e["plate"], (tx, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(panel, f"{e['time'].split(' ')[1]}   {e['conf']}%", (tx, y + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)
        cv2.line(panel, (0, y + ROW_H - 1), (PANEL_W, y + ROW_H - 1), (55, 55, 55), 1)
        y += ROW_H
    if not logs:
        cv2.putText(panel, "Engin numer enn", (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
    return panel
