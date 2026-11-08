# Númeraplata (Numby ALPR)

Léttur íslenskur númeraplötulesari (ALPR) í tveimur útgáfum:
1. **Android app (`.apk`)** — keyrir 100% á símanum án nets (CameraX + Google ML Kit).
2. **Python útgáfa (`alpr.py`)** — keyrir á tölvu með vefmyndavél, IP/RTSP símastraumi eða innbyggðu vefviðmóti (`fast-alpr` ONNX + OpenCV).

Báðar útgáfur:
- Þekkja íslensk bílnúmer (`AB123` og `ABC12`), hunsa skoðunarmiða í miðjunni (`LO 25 L46` → `LOL46`) og leiðrétta algengar OCR-villur (`0`↔`O`, `1`↔`I`, `8`↔`B`).
- Sameina nálægar aflestrar af sama bíl (`VSO07` / `XLS53` → `VSD07` / `IXL53`) innan 25 sekúndna glugga og uppfæra í skarpasta rammann.
- Flokka atvik eftir dögum og tímaröð, vista myndir í fullri upplausn með grænum ramma utan um plötuna og bjóða upp á deilingu (CSV og mynd).

---

## 1. Python útgáfa (`alpr.py`)

### Uppsetning
```bash
pip install -r requirements.txt
```

### Keyrsla
```bash
# 1) Beint af vefmyndavél (gluggi + vistar í plates.csv og captures/):
python alpr.py

# 2) Vefviðmót (fyrir síma eða vafra á http://localhost:8000):
python alpr.py --web

# 3) Beinn straumur úr síma (t.d. IP Webcam / RTSP):
python alpr.py http://192.168.1.50:8080/video

# Keyra innbyggð próf:
python alpr.py --test
```

---

## 2. Android útgáfa (`app/`)

Tilbúna `.apk` skrá má sækja beint undir **[Releases](https://github.com/robertbeck-stack/numeraplata/releases)** eða smíða úr grunni:

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```
