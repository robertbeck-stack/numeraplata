"""Numby ALPR package."""
__version__ = "v2.3.0"
VERSION = __version__

from .plates import valid_plate, is_same_car, PLATE_RE, STICKER_RE
from .scanner import PlateScanner, make_alpr, SCAN_MODEL, OCR_MODEL
from .objects import detect_objects, detect_vehicles, plate_on_vehicle, VEHICLE_LABELS, get_vehicle_detector
from .colors import appearance, scene_cast, color_stats, color_name, similar
from .tracking import BoxTracker
from .traffic import VehicleTracks, NightCounter, VEHICLE_NAMES, COUNT_LINE
from .people import PeopleCounter
from .storage import (
    load_logs, save_logs, delete_entry, clamp_zoom, crop_zoom,
    save_snapshot, is_safe_path, prune_captures,
    LOG_FILE, TRAFFIC_FILE, PEOPLE_FILE, CAPTURES_DIR, SETTINGS_FILE
)
from .drawing import draw_vehicles, draw_hidden, draw_people, draw_results, build_panel, clean_text_for_opencv

__all__ = [
    "__version__",
    "VERSION",
    "valid_plate",
    "is_same_car",
    "PLATE_RE",
    "STICKER_RE",
    "PlateScanner",
    "make_alpr",
    "SCAN_MODEL",
    "OCR_MODEL",
    "detect_objects",
    "detect_vehicles",
    "plate_on_vehicle",
    "VEHICLE_LABELS",
    "get_vehicle_detector",
    "appearance",
    "scene_cast",
    "color_stats",
    "color_name",
    "similar",
    "BoxTracker",
    "VehicleTracks",
    "NightCounter",
    "VEHICLE_NAMES",
    "COUNT_LINE",
    "PeopleCounter",
    "load_logs",
    "save_logs",
    "delete_entry",
    "clamp_zoom",
    "crop_zoom",
    "save_snapshot",
    "is_safe_path",
    "prune_captures",
    "LOG_FILE",
    "TRAFFIC_FILE",
    "PEOPLE_FILE",
    "CAPTURES_DIR",
    "SETTINGS_FILE",
    "draw_vehicles",
    "draw_hidden",
    "draw_people",
    "draw_results",
    "build_panel",
    "clean_text_for_opencv",
]
