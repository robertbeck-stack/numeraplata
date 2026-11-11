"""Rökfræði fyrir íslensk bílnúmer og samanburð á plötum."""
import re

PLATE_RE = re.compile(r"^([A-Z]{2}[0-9]{3}|[A-Z]{3}[0-9]{2})$")
# Styður bæði LL-skoðunarmiði-DDD og LLL-skoðunarmiði-DD
STICKER_RE = re.compile(r"^([A-Z0-9]{2})[0-9]{2}([A-Z0-9]{3})$|^([A-Z0-9]{3})[0-9]{2}([A-Z0-9]{2})$")

TO_LET = {"0": "O", "1": "I", "8": "B"}
TO_DIG = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "B": "8", "S": "5"}

# Algeng stafavíxl í OCR á númeraplötum
CONFUSIONS = {
    frozenset(("0", "O")), frozenset(("0", "Q")), frozenset(("0", "D")), frozenset(("O", "D")),
    frozenset(("1", "I")), frozenset(("1", "L")), frozenset(("1", "T")), frozenset(("I", "L")),
    frozenset(("8", "B")), frozenset(("8", "3")), frozenset(("B", "3")),
    frozenset(("5", "S")), frozenset(("2", "Z")), frozenset(("6", "G")),
    frozenset(("V", "U")), frozenset(("V", "Y")), frozenset(("U", "Y")),
    frozenset(("E", "F")), frozenset(("P", "R")), frozenset(("K", "X")),
    frozenset(("C", "G")), frozenset(("M", "N")),
}


def _normalize_five(s: str) -> str:
    """Leiðréttir stafafjölda í 5 stafa íslensku númeri eftir staðsetningu bókstafa og talna."""
    if len(s) != 5:
        return s
    if not (s[0].isalpha() or s[1].isalpha()):
        return s
    if not (s[3].isdigit() or s[4].isdigit()):
        return s
    return (
        TO_LET.get(s[0], s[0])
        + TO_LET.get(s[1], s[1])
        + s[2]
        + TO_DIG.get(s[3], s[3])
        + TO_DIG.get(s[4], s[4])
    )


def valid_plate(raw: str) -> str | None:
    """Athugar hvort texti sé gilt íslenskt bílnúmer og skilar stöðluðu númeri eða None."""
    cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
    # "IS" fremst er landsmerkið á bláu röndinni, en aðeins ef eitthvað heilt númer er eftir
    candidates = ([cleaned[2:]] if cleaned.startswith("IS") and len(cleaned) > 5 else []) + [cleaned]
    for c in candidates:
        norm5 = _normalize_five(c)
        if PLATE_RE.match(norm5):
            return norm5
        m = STICKER_RE.match(c)
        if m:
            g1 = m.group(1) or m.group(3)
            g2 = m.group(2) or m.group(4)
            combined = _normalize_five(g1 + g2)
            if PLATE_RE.match(combined):
                return combined
    return None


def is_same_car(a: str, b: str) -> bool:
    """Metur hvort tveir lestrar séu af sama bíl.

    Krefst þess að 1 stafs munur sé líklegur OCR villulestur (t.d. O/D eða 8/B),
    eða hliðrun vegna plöturamma/fána (t.d. I fremst), svo ólíkir bílar í röð
    (eins og WA936 og WA937) séu ekki taldir sami bíllinn.
    """
    if len(a) != 5 or len(b) != 5:
        return False
    if a == b:
        return True

    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diffs) == 1 and frozenset(diffs[0]) in CONFUSIONS:
        return True

    dp = [[0] * 6 for _ in range(6)]
    for i in range(1, 6):
        for j in range(1, 6):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])

    if dp[5][5] >= 4:
        # Hliðrun vegna gervistafs við jaðar (t.d. bláa röndin eða rammi lesinn sem I/1/L)
        if a[0] in "I1L" or b[0] in "I1L" or a[-1] in "I1L" or b[-1] in "I1L":
            return True

    return False
