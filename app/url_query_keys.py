"""URL query-key ambiguity auditing.

Splits the query component of a raw URL into parameters using the
**undecoded** ``&`` and ``=`` characters and keeps each key's start/end
codepoint offsets in the original URL.  Every key is then percent-decoded
exactly once (``+`` decodes to a space, ``%XX`` hex triplets decode to
UTF-8) and Unicode NFKC normalised.  When two or more parameter keys share
the same normalised form, their collision group is reported with every
member's raw key, decoded key, 1-based parameter number and original
position.  Parameter values never take part in decoding or comparison.

Invalid percent encoding (a ``%`` not followed by two hex digits) or a
percent-decoded byte run that is not valid UTF-8 raises
:class:`QueryKeyParseError` carrying the offending span's offsets in the
original URL; callers reject the whole request, so no partial report is
ever emitted.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from .json_audit import line_col

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


class QueryKeyParseError(ValueError):
    """A query key cannot be percent/UTF-8 decoded.

    ``error`` is ``invalid_percent_encoding`` or ``invalid_utf8``;
    ``start`` / ``end`` are 0-based codepoint offsets in the original
    URL covering the offending unit (``end`` exclusive).
    """

    def __init__(
        self,
        error: str,
        start: int,
        end: int,
        message: str,
    ):
        super().__init__(f"offset {start}: {message}")
        self.error = error
        self.start = start
        self.end = end
        self.message = message


# ---------------------------------------------------------------------------
# Parsed model
# ---------------------------------------------------------------------------

@dataclass
class QueryKey:
    """One parameter key after the single percent/UTF-8 decode pass."""

    index: int            # 1-based parameter number in the query string
    raw_key: str          # key verbatim, before any decoding
    decoded_key: str      # percent-decoded key (+ -> space), not NFKC-ed
    normalised_key: str   # NFKC normalisation of the decoded key
    start: int            # codepoint offset of the key in the URL
    end: int              # exclusive codepoint offset of the key


@dataclass
class QueryKeyCollision:
    """Two or more parameter keys sharing one NFKC normalised form."""

    normalised: str
    members: list[QueryKey] = field(default_factory=list)

    def to_dict(self, url: str) -> dict:
        return {
            "normalised": self.normalised,
            "count": len(self.members),
            "members": [_key_to_dict(key, url) for key in self.members],
        }


@dataclass
class QueryKeyReport:
    parameter_count: int
    collisions: list[QueryKeyCollision] = field(default_factory=list)

    def to_dict(self, url: str) -> dict:
        return {
            "stored": False,
            "url": url,
            "url_length": len(url),
            "parameter_count": self.parameter_count,
            "collision_count": len(self.collisions),
            "collisions": [
                collision.to_dict(url) for collision in self.collisions
            ],
        }


def _key_to_dict(key: QueryKey, url: str) -> dict:
    line, column = line_col(url, key.start)
    return {
        "index": key.index,
        "raw_key": key.raw_key,
        "decoded_key": key.decoded_key,
        "normalised_key": key.normalised_key,
        "start": key.start,
        "end": key.end,
        "position": {"line": line, "column": column},
    }


# ---------------------------------------------------------------------------
# Query extraction / splitting (always on the undecoded text)
# ---------------------------------------------------------------------------

def _query_span(url: str) -> tuple[int, str] | None:
    """Return (absolute start offset, query text) of the URL's query.

    The query starts right after the first ``?`` and ends at the first
    ``#`` that follows it (the fragment).  A URL without ``?`` has no
    query component.
    """
    qmark = url.find("?")
    if qmark < 0:
        return None
    start = qmark + 1
    fragment = url.find("#", start)
    end = fragment if fragment >= 0 else len(url)
    return start, url[start:end]


def _decode_key(raw_key: str, base: int) -> str:
    """Percent-decode one raw key exactly once, with ``+`` as space.

    ``base`` is the URL offset of the key's first character.  Each
    maximal run of ``%XX`` triplets is decoded as one UTF-8 byte run;
    literal characters (including non-ASCII ones) are kept verbatim and
    terminate a run.  Raises :class:`QueryKeyParseError` on a malformed
    triplet or an invalid UTF-8 run, spanning the offending source unit.
    """
    decoded: list[str] = []
    byte_run = bytearray()
    run_start = -1          # URL offset of the run's first '%'
    run_end = -1            # exclusive URL offset after the run's last unit
    n = len(raw_key)
    i = 0

    def flush() -> None:
        nonlocal byte_run, run_start, run_end
        if not byte_run:
            return
        try:
            decoded.append(byte_run.decode("utf-8"))
        except UnicodeDecodeError:
            raise QueryKeyParseError(
                "invalid_utf8",
                run_start,
                run_end,
                "percent-decoded bytes are not valid UTF-8",
            )
        byte_run = bytearray()
        run_start = run_end = -1

    while i < n:
        ch = raw_key[i]
        url_offset = base + i
        if ch == "%":
            if i + 2 >= n or any(
                raw_key[j] not in _HEX_DIGITS for j in (i + 1, i + 2)
            ):
                # Malformed unit: '%' at the end, '%X' or '%XY' with a
                # non-hex character; span the characters actually present.
                bad_end = min(i + 3, n)
                raise QueryKeyParseError(
                    "invalid_percent_encoding",
                    url_offset,
                    base + bad_end,
                    "'%' must be followed by two hexadecimal digits",
                )
            if not byte_run:
                run_start = url_offset
            byte_run.append(int(raw_key[i + 1:i + 3], 16))
            run_end = url_offset + 3
            i += 3
            continue
        # A literal character ends the current percent-decoded byte run.
        flush()
        decoded.append(" " if ch == "+" else ch)
        i += 1

    flush()
    return "".join(decoded)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_query_keys(url: str) -> QueryKeyReport:
    """Split ``url``'s query and report NFKC key collisions.

    Raises :class:`QueryKeyParseError` before returning whenever any key
    fails the single percent/UTF-8 decode pass, so a malformed URL never
    yields a partial report.  Nothing is persisted.
    """
    span = _query_span(url)
    if span is None:
        return QueryKeyReport(parameter_count=0)
    query_start, query = span

    # Split on the undecoded '&' (str.split keeps empty parameters, so a
    # doubled or trailing '&' yields an empty key) and take the substring
    # before the first undecoded '=' as the key; further '=' characters
    # belong to the value, which is never inspected.
    keys: list[QueryKey] = []
    cursor = query_start
    for index, parameter in enumerate(query.split("&"), start=1):
        eq = parameter.find("=")
        raw_key = parameter if eq < 0 else parameter[:eq]
        key_start = cursor
        key_end = key_start + len(raw_key)
        decoded = _decode_key(raw_key, key_start)
        keys.append(
            QueryKey(
                index=index,
                raw_key=raw_key,
                decoded_key=decoded,
                normalised_key=unicodedata.normalize("NFKC", decoded),
                start=key_start,
                end=key_end,
            )
        )
        cursor += len(parameter) + 1  # +1 for the consumed '&'

    groups: dict[str, QueryKeyCollision] = {}
    order: list[str] = []
    for key in keys:
        group = groups.get(key.normalised_key)
        if group is None:
            group = QueryKeyCollision(normalised=key.normalised_key)
            groups[key.normalised_key] = group
            order.append(key.normalised_key)
        group.members.append(key)

    collisions = [groups[name] for name in order if len(groups[name].members) >= 2]
    return QueryKeyReport(
        parameter_count=len(keys),
        collisions=collisions,
    )
