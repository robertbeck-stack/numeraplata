"""Numby ALPR: Íslenskur númeraplötulesari fyrir vefmyndavél, RTSP/símastraum og vefviðmót."""
import argparse
import sys
import time

# Tryggja rétta UTF-8 kóðun á Windows stjórnborði
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from numby import (
    valid_plate, is_same_car, PLATE_RE, STICKER_RE,
    PlateScanner, make_alpr, SCAN_MODEL, OCR_MODEL,
    detect_objects, detect_vehicles, plate_on_vehicle, VEHICLE_LABELS, get_vehicle_detector,
    appearance, scene_cast, color_stats, color_name, similar,
    BoxTracker,
    VehicleTracks, NightCounter, VEHICLE_NAMES, COUNT_LINE,
    PeopleCounter,
    load_logs, save_logs, delete_entry, clamp_zoom, crop_zoom,
    save_snapshot, is_safe_path, prune_captures,
    LOG_FILE, TRAFFIC_FILE, PEOPLE_FILE, CAPTURES_DIR, SETTINGS_FILE,
    draw_vehicles, draw_hidden, draw_people, draw_results, build_panel, clean_text_for_opencv,
    VERSION, __version__
)
from numby import pipeline
from numby.server import serve, HTML, DASHBOARD_HTML
from numby.dashboard import dashboard, run, WINDOW, DISPLAY_MAX_H

# Bakábendingar fyrir prófanir og einingapróf
VEHICLE_CHECK = pipeline.VEHICLE_CHECK
RESIGHT_SECONDS = pipeline.RESIGHT_SECONDS
_pending = pipeline._pending
_rejected = pipeline._rejected


def process_frame(alpr_instance, frame, logs: list[dict], results=None) -> list[dict]:
    """Vinnur úr ramma. Nýtir staðværar breytur (time, _pending, VEHICLE_CHECK) fyrir prófanir."""
    return pipeline.process_frame(
        alpr_instance, frame, logs, results=results,
        time_module=time, pending_dict=_pending,
        vehicle_check=VEHICLE_CHECK, resight_sec=RESIGHT_SECONDS
    )


def main():
    # Innri prófanir á rökfræði
    assert valid_plate("AB-123") == "AB123"
    assert valid_plate("IS LO 25 L46") == "LOL46"
    assert valid_plate("L0 25 L46") == "LOL46"
    assert valid_plate("AB 12O") == "AB120"
    assert valid_plate("PARIS") is None
    assert valid_plate("x") is None
    assert valid_plate("IS123") == "IS123"
    assert valid_plate("IS AB 123") == "AB123"
    assert valid_plate("ISA12") == "ISA12"
    assert is_same_car("VSD07", "VSO07")
    assert is_same_car("XLS53", "IXL53")
    assert not is_same_car("VSD07", "ABC12")

    parser = argparse.ArgumentParser(description="Numby ALPR: Íslenskur númeraplötulesari og umferðarteljari")
    parser.add_argument("source", nargs="?", default="0", help="Myndavélarnúmer (t.d. 0), RTSP straumur eða myndbandsskrá")
    parser.add_argument("--host", default="127.0.0.1", help="Veffang til að hlusta á (sjálfgefið 127.0.0.1, notaðu 0.0.0.0 fyrir staðarnet)")
    parser.add_argument("--port", type=int, default=8000, help="Gátt fyrir vefviðmót (sjálfgefið 8000)")
    parser.add_argument("--window", action="store_true", help="Sýna mynd í OpenCV glugga með hliðarstiku")
    parser.add_argument("--web", action="store_true", help="Vefviðmót fyrir vefmyndavél í síma/vafra")
    parser.add_argument("--test", action="store_true", help="Keyra sjálfsprófun og hætta")
    parser.add_argument("--prune", action="store_true", help="Hreinsa gamlar myndir í captures/ möppu")

    args = parser.parse_args()

    if args.prune:
        count = prune_captures(max_days=30)
        print(f"Hreinsun lokið: {count} myndum eytt úr {CAPTURES_DIR}/")
        return

    if args.test:
        print("self-check ok")
        return

    if args.web:
        serve(port=args.port, host=args.host if args.host != "127.0.0.1" else "0.0.0.0")
        return

    src = int(args.source) if args.source.isdigit() else args.source
    if args.window:
        dashboard(src, host=args.host, port=args.port, window=True)
    else:
        dashboard(src, host=args.host, port=args.port, window=False)


if __name__ == "__main__":
    main()
