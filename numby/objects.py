"""Greining á bílum og fólki með RF-DETR líkani."""
import cv2
from types import SimpleNamespace

VEHICLE_MODEL = "rf-detr-nano-384-coco"
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}
VEHICLE_THREADS = 2
VEHICLE_INTERVAL = 1.0
PERSON_MIN_CONF = 0.5
_vehicle_det = None


def get_vehicle_detector():
    """Skilar samnýttu eintaki af bíla- og fólksgreininum."""
    global _vehicle_det
    if _vehicle_det is None:
        from open_image_models.detection.factory import create_detector
        from .scanner import _ort_opts
        _vehicle_det = create_detector(VEHICLE_MODEL, conf_thresh=0.3, sess_options=_ort_opts(VEHICLE_THREADS))
    return _vehicle_det


def detect_vehicles(frame) -> list:
    """Skilar aðeins greindum ökutækjum úr ramma."""
    return [r for r in detect_objects(frame) if r.label in VEHICLE_LABELS]


def detect_objects(frame) -> list:
    """Greinir bíla og fólk. Myndin er minnkuð fyrst (líkanið notar hvort eð er 384px), sem sparar
    forvinnslu í Python og heldur plötugreiningunni hraðri. Kassar eru skalaðir aftur."""
    h, w = frame.shape[:2]
    s = min(1.0, 640 / max(h, w))
    small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
    out = []
    for r in get_vehicle_detector().predict(small):
        if r.label not in VEHICLE_LABELS and not (r.label == "person" and r.confidence >= PERSON_MIN_CONF):
            continue
        b = r.bounding_box
        box = SimpleNamespace(x1=b.x1 / s, y1=b.y1 / s, x2=b.x2 / s, y2=b.y2 / s)
        out.append(SimpleNamespace(label=r.label, confidence=r.confidence, bounding_box=box))
    kept = []
    for o in sorted(out, key=lambda o: -o.confidence):
        b = o.bounding_box
        for q in kept:
            if (q.label == "person") != (o.label == "person"):
                continue
            c = q.bounding_box
            ix, iy = min(b.x2, c.x2) - max(b.x1, c.x1), min(b.y2, c.y2) - max(b.y1, c.y1)
            if ix > 0 and iy > 0:
                inter, a1, a2 = ix * iy, (b.x2 - b.x1) * (b.y2 - b.y1), (c.x2 - c.x1) * (c.y2 - c.y1)
                iou = inter / (a1 + a2 - inter)
                if iou > 0.5 or (q.label == o.label and inter > 0.75 * min(a1, a2)):
                    break  # sami hlutur greindur tvisvar
        else:
            kept.append(o)
    return kept


def plate_on_vehicle(frame, box) -> bool:
    """Sannreynir hvort plötukassi sé innan í einhverju greindu ökutæki."""
    if box is None:
        return False
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    for r in detect_vehicles(frame):
        v = r.bounding_box
        pad_x, pad_y = (v.x2 - v.x1) * 0.05, (v.y2 - v.y1) * 0.05
        if v.x1 - pad_x <= cx <= v.x2 + pad_x and v.y1 - pad_y <= cy <= v.y2 + pad_y:
            return True
    return False
