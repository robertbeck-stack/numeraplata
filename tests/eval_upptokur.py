"""Mat á bílatalningu á raunverulegum upptökum úr myndavélinni (þarf líkönin, tekur ~1 mínútu).
Keyrsla:  python tests/eval_upptokur.py

Upptökurnar eru í tests/data (ekki í git):
  gata_rokkur_2026-11-10/   80 rammar, 2 á sekúndu, í rökkri. Rétt svar: 6 bílar til hægri yfir talningarlínuna.
  gata_nott_2026-11-11.mp4  100 sekúndur í myrkri (úr lifandi straumnum). Rétt svar: um 22-25 bílar, flestir til vinstri.
Réttu svörin eru talin af myndunum með augum."""
import glob
import os
import sys
import tempfile

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import alpr

os.chdir(tempfile.mkdtemp())

tr = alpr.VehicleTracks()
for f in sorted(glob.glob(os.path.join(HERE, "data", "gata_rokkur_2026-11-10", "f*.jpg"))):
    img = cv2.imread(f)
    tr.update(alpr.detect_objects(img), float(os.path.basename(f)[4:9]), img.shape[1], img)
print(f"rökkur: ← {tr.left}  → {tr.right}   (rétt: ← 0  → 6)")

counted = []
nc = alpr.NightCounter(lambda label, direction: counted.append(direction))
cap = cv2.VideoCapture(os.path.join(HERE, "data", "gata_nott_2026-11-11.mp4"))
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    nc.update(frame, i / 26.06)
    i += 1
print(f"nótt:   ← {counted.count('vinstri')}  → {counted.count('hægri')}   (rétt: samtals um 22-25, þar af 5-6 til hægri)")
