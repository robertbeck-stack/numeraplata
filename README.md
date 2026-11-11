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
# 1) Vefmyndavél með stjórnborði í vafra (http://127.0.0.1:8000):
python alpr.py

# 2) Sama mynd í OpenCV-glugga í stað vafra:
python alpr.py --window

# 3) Beinn straumur úr síma (t.d. IP Webcam / RTSP) eða myndbandsskrá:
python alpr.py http://192.168.1.50:8080/video

# 4) Myndavél símans í gegnum vafra (síminn sendir ramma á tölvuna):
python alpr.py --web

# Prófanir:
python alpr.py --test
python tests/test_logic.py
```

### Hvað stjórnborðið gerir
- **Númer:** les plötur og skráir í `plates.csv` með mynd í `captures/`. Myndinni er skipt í flísar svo litlar plötur finnist, og númer þarf að lesast eins þrisvar áður en það er skráð.
- **Bílar:** kassi utan um hvern bíl með tegund, lit og stefnuör. Hreyfirakning á hverjum ramma lætur kassann fylgja bílum á ferð.
- **Talning:** bílar sem fara yfir talningarlínuna (hvít strikalína) eru taldir eftir stefnu í `traffic.csv`. Í myrkri er talið eftir fram- og afturljósum.
- **Fólk:** gulur kassi í lifandi mynd og fjöldi á mínútu í `people.csv`. Engar myndir af fólki eru vistaðar sjálfkrafa.
- **Annað:** stafrænn aðdráttur, eyðing færslna og „Taka mynd“ sem vistar ramma með öllum kössum.

Kóðinn er í `numby/`; `alpr.py` er inngangurinn.

---

## 2. Android útgáfa (`app/`)

Tilbúna `.apk` skrá má sækja beint undir **[Releases](https://github.com/robertbeck-stack/numeraplata/releases)** eða smíða úr grunni:

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```
