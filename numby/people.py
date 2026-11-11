"""Fólkstalning og skráning á hámarksfjölda á mínútu."""
from datetime import datetime
import os
import threading

PEOPLE_FILE = "people.csv"  # aðeins tími og fjöldi, engar myndir af fólki


class PeopleCounter:
    """Skráir mesta fjölda fólks á hverri mínútu í people.csv (aðeins mínútur þar sem einhver sást)."""

    def __init__(self, path: str = PEOPLE_FILE):
        self.path = path
        self.now = 0
        self.minute = None
        self.minute_max = 0
        self.peak = 0
        self.peak_time = ""
        self.today = datetime.now().strftime("%Y-%m-%d")
        self._lock = threading.Lock()
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                next(f, None)
                for line in f:
                    p = line.strip().split(",")
                    if len(p) == 3 and p[0] == self.today and p[2].isdigit() and int(p[2]) > self.peak:
                        self.peak = int(p[2])
                        self.peak_time = p[1]

    def update(self, count: int) -> None:
        with self._lock:
            now = datetime.now()
            day, minute = now.strftime("%Y-%m-%d"), now.strftime("%H:%M")
            if day != self.today:
                self.today = day
                self.peak = 0
                self.peak_time = ""
            if (day, minute) != self.minute:
                self._flush_unlocked()
                self.minute = (day, minute)
                self.minute_max = 0
            self.now = count
            self.minute_max = max(self.minute_max, count)
            if count > self.peak:
                self.peak = count
                self.peak_time = minute

    def flush(self) -> None:
        """Skrifar út gögn núverandi mínútu, t.d. þegar forritið stöðvast."""
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if self.minute is None or self.minute_max == 0:
            return
        new = not os.path.isfile(self.path)
        with open(self.path, "a", encoding="utf-8") as f:
            if new:
                f.write("Dagur,Tími,Fjöldi\n")
            f.write(f"{self.minute[0]},{self.minute[1]},{self.minute_max}\n")
