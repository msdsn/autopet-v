#!/usr/bin/env python3
"""
Early access to a still-downloading ZIP archive.

The autoPET V dataset ships as one 150 GB zip that takes ~2 h to fetch. This module
hands zipfile a file-like object stitched from byte ranges of the virtual full
archive: the growing partial file, tail.bin (the central directory, pulled up front
with an HTTP range request), and any number of out-of-order patch ranges. A read
that falls into a hole raises GapError.

Archive layout, from the central directory. PSMA CT lands only after all PSMA PET
and the labels sit at the very end, so both are worth pulling early (--fetch-*):

    entries    0 -    4   metadata (dataset.json, fingerprint, splits, 2 csv)
    entries    5 -  601   imagesTr/psma_*_0001.nii.gz     (PET, 597 cases)
    entries  602 - 1198   imagesTr/psma_*_0000.nii.gz     (CT,  597 cases)
    entries 1199 - 3226   imagesTr/fdg_*_000{0,1}.nii.gz  (interleaved, 1014 cases)
    entries 3227 - 4837   labelsTr/*.nii.gz               (1611 labels, ~0.26 GB)

Usage:

    python partial_zip_extract.py layout
    python partial_zip_extract.py meta   --out /content/drive/MyDrive/autoPET/meta
    python partial_zip_extract.py avail  --json /content/work/avail.json
    python partial_zip_extract.py labels --out /content/drive/MyDrive/autoPET/labelsTr
    python partial_zip_extract.py cases  --n 10 --out /content/work/testcase
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------- constants --

ZIP_URL = ("https://fdat.uni-tuebingen.de/records/rdkqd-wdh87/files/"
           "psma-fdg-pet-ct-lesions_v2.zip?download=1")
ZIP_SIZE = 150_293_961_309
DATA_DIR = Path(os.environ.get("AUTOPET_DATA", "/content/data"))
ZIP_PATH = DATA_DIR / "psma-fdg.zip"
TAIL_PATH = DATA_DIR / "tail.bin"
TAIL_LEN = 8_000_000
PATCH_DIR = DATA_DIR / "patch"
PATCH_MANIFEST = PATCH_DIR / "patches.json"
TOPDIR = "PSMA-FDG-PET-CT-Lesions_v2"
META_FILES = ["dataset.json", "dataset_fingerprint.json", "splits_final.json",
              "fdg_metadata.csv", "psma_metadata.csv"]


class GapError(IOError):
    """A read fell into a not-yet-downloaded region of the archive."""


# ------------------------------------------------------------ the file-like --

@dataclass(frozen=True)
class Segment:
    start: int          # offset in the virtual full archive
    length: int
    path: Path
    file_offset: int = 0   # offset of `start` inside `path`

    @property
    def end(self) -> int:
        return self.start + self.length


class PartialArchive(io.RawIOBase):
    """Read-only file-like view over the assembled byte ranges of the archive.

    Implements just the read/seek/tell/seekable that zipfile needs. The size of the
    growing main file is re-stat'd every `refresh_every` seconds so a long-lived
    instance sees newly downloaded bytes.
    """

    def __init__(self, size: int = ZIP_SIZE, main: Path | None = ZIP_PATH,
                 tail: Path | None = TAIL_PATH, tail_len: int = TAIL_LEN,
                 patches: Sequence[Segment] = (), refresh_every: float = 2.0):
        super().__init__()
        self.size = size
        self._pos = 0
        self._main = Path(main) if main else None
        self._refresh_every = refresh_every
        self._last_stat = 0.0
        self._main_len = 0
        self._handles: dict[Path, io.BufferedReader] = {}
        self._static: list[Segment] = []
        if tail is not None and Path(tail).exists():
            n = min(tail_len, Path(tail).stat().st_size)
            self._static.append(Segment(size - n, n, Path(tail)))
        self._static.extend(patches)
        self._static.sort(key=lambda s: s.start)
        self._refresh()

    # -- plumbing ----------------------------------------------------------
    def _refresh(self) -> None:
        if self._main is None:
            return
        now = time.time()
        if now - self._last_stat < self._refresh_every and self._main_len:
            return
        self._last_stat = now
        try:
            self._main_len = self._main.stat().st_size
        except FileNotFoundError:
            self._main_len = 0

    @property
    def downloaded(self) -> int:
        self._refresh()
        return self._main_len

    def _handle(self, path: Path) -> io.BufferedReader:
        h = self._handles.get(path)
        if h is None:
            h = open(path, "rb", buffering=1024 * 1024)
            self._handles[path] = h
        return h

    def _segment_at(self, pos: int) -> Segment | None:
        self._refresh()
        if self._main is not None and pos < self._main_len:
            return Segment(0, self._main_len, self._main)
        for s in self._static:
            if s.start <= pos < s.end:
                return s
        return None

    def available(self, start: int, length: int) -> bool:
        """True if [start, start+length) is fully covered, spanning segments.

        Segments may abut (the patched labelsTr range runs into tail.bin), so walk
        forward rather than requiring a single segment to hold the whole range.
        """
        pos = start
        end = start + length
        while pos < end:
            s = self._segment_at(pos)
            if s is None:
                return False
            pos = s.end
        return True

    # -- io interface ------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if self._pos < 0:
            raise ValueError("negative seek position")
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        if n == 0 or self._pos >= self.size:
            return b""
        n = min(n, self.size - self._pos)
        out = bytearray()
        while n > 0:
            seg = self._segment_at(self._pos)
            if seg is None:
                raise GapError(
                    f"offset {self._pos} ({self._pos / 1e9:.2f} GB) is not "
                    f"downloaded yet (have {self.downloaded / 1e9:.2f} GB)")
            take = min(n, seg.end - self._pos)
            h = self._handle(seg.path)
            h.seek(seg.file_offset + (self._pos - seg.start))
            chunk = h.read(take)
            if not chunk:
                raise GapError(f"short read at {self._pos}")
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:  # pragma: no cover - zipfile uses read()
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()
        super().close()


# ------------------------------------------------------------ patch handling --

def load_patches() -> list[Segment]:
    if not PATCH_MANIFEST.exists():
        return []
    raw = json.loads(PATCH_MANIFEST.read_text())
    return [Segment(int(p["start"]), int(p["length"]), Path(p["path"]),
                    int(p.get("file_offset", 0))) for p in raw]


def save_patches(patches: Iterable[Segment]) -> None:
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_MANIFEST.write_text(json.dumps(
        [{"start": s.start, "length": s.length, "path": str(s.path),
          "file_offset": s.file_offset} for s in patches], indent=1))


def fetch_range(start: int, end_incl: int, dest: Path, tries: int = 5) -> None:
    """Fetch bytes [start, end_incl] of the archive with curl into dest.

    The fdat URL 302-redirects to a presigned S3 URL valid for 60 s; curl -L
    re-issues the ranged GET against the redirect target, which supports ranges.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    want = end_incl - start + 1
    for attempt in range(1, tries + 1):
        cmd = ["curl", "-sSL", "--fail", "--retry", "3", "--retry-delay", "5",
               "--max-time", "3600", "-r", f"{start}-{end_incl}",
               "-o", str(dest), ZIP_URL]
        rc = subprocess.call(cmd)
        got = dest.stat().st_size if dest.exists() else 0
        if rc == 0 and got == want:
            return
        print(f"  range fetch attempt {attempt} failed (rc={rc}, got={got}/{want})",
              file=sys.stderr)
        time.sleep(5)
    raise IOError(f"could not fetch range {start}-{end_incl}")


def add_patch(start: int, end_incl: int, name: str) -> Segment:
    """Fetch a range and register it so PartialArchive can serve it."""
    dest = PATCH_DIR / f"{name}.bin"
    length = end_incl - start + 1
    patches = load_patches()
    for p in patches:
        if p.start == start and p.length == length and p.path.exists():
            print(f"  patch {name} already present")
            return p
    print(f"  fetching {name}: {length / 1e6:.1f} MB @ {start}")
    t0 = time.time()
    fetch_range(start, end_incl, dest)
    seg = Segment(start, length, dest)
    patches = [p for p in patches if not (p.start == start and p.length == length)]
    patches.append(seg)
    save_patches(patches)
    print(f"  done in {time.time() - t0:.0f}s ({length / 1e6 / max(time.time() - t0, 1e-9):.1f} MB/s)")
    return seg


# ------------------------------------------------------------------ catalog --

@dataclass
class Entry:
    name: str
    header_offset: int
    compress_size: int
    file_size: int
    compress_type: int

    # The local file header is 30 fixed bytes + filename + extra field, and the
    # central directory does not record the local extra length -- so bound it with
    # slack until the header itself is readable and Catalog.exact_end can parse it.
    def end_offset(self, slack: int = 65536 + 128) -> int:
        return self.header_offset + 30 + len(self.name.encode()) + slack + self.compress_size


class Catalog:
    """The archive's central directory + convenience views over it."""

    def __init__(self, archive: PartialArchive):
        self.archive = archive
        self.zf = zipfile.ZipFile(archive)
        self.entries = [Entry(i.filename, i.header_offset, i.compress_size,
                              i.file_size, i.compress_type)
                        for i in self.zf.infolist()]
        self.by_name = {e.name: e for e in self.entries}
        self._end_cache: dict[int, int] = {}

    # -- views -------------------------------------------------------------
    def meta(self) -> list[Entry]:
        return [self.by_name[f"{TOPDIR}/{m}"] for m in META_FILES
                if f"{TOPDIR}/{m}" in self.by_name]

    def images(self) -> list[Entry]:
        return [e for e in self.entries if e.name.startswith(f"{TOPDIR}/imagesTr/")]

    def labels(self) -> list[Entry]:
        return [e for e in self.entries if e.name.startswith(f"{TOPDIR}/labelsTr/")]

    def case_ids(self) -> list[str]:
        return sorted(Path(e.name).name[:-len(".nii.gz")] for e in self.labels())

    def case_entries(self, case: str) -> dict[str, Entry | None]:
        return {
            "ct": self.by_name.get(f"{TOPDIR}/imagesTr/{case}_0000.nii.gz"),
            "pet": self.by_name.get(f"{TOPDIR}/imagesTr/{case}_0001.nii.gz"),
            "label": self.by_name.get(f"{TOPDIR}/labelsTr/{case}.nii.gz"),
        }

    def region(self, entries: Sequence[Entry]) -> tuple[int, int]:
        """Byte range [start, end_incl] that covers all of entries."""
        lo = min(e.header_offset for e in entries)
        hi = max(e.end_offset(slack=0) for e in entries)
        return lo, min(hi + 4096, ZIP_SIZE - 1)

    # -- availability ------------------------------------------------------
    def exact_end(self, e: Entry) -> int:
        """Exclusive end offset of the member's data, parsing the local header.

        Falls back to the pessimistic 64 KiB slack when the header itself is not
        downloaded yet, in which case the member is not available anyway.
        """
        cached = self._end_cache.get(e.header_offset)
        if cached is not None:
            return cached
        if not self.archive.available(e.header_offset, 30):
            return e.end_offset()
        self.archive.seek(e.header_offset)
        hdr = self.archive.read(30)
        if hdr[:4] != b"PK\x03\x04":
            return e.end_offset()
        name_len = int.from_bytes(hdr[26:28], "little")
        extra_len = int.from_bytes(hdr[28:30], "little")
        end = e.header_offset + 30 + name_len + extra_len + e.compress_size
        self._end_cache[e.header_offset] = end
        return end

    def is_available(self, e: Entry | None) -> bool:
        if e is None:
            return False
        end = self.exact_end(e)
        return self.archive.available(e.header_offset, end - e.header_offset)

    def available_cases(self) -> dict[str, list[str]]:
        out = {"complete": [], "ct_only": [], "pet_only": [], "label_only": [],
               "none": []}
        for c in self.case_ids():
            ent = self.case_entries(c)
            ct = self.is_available(ent["ct"])
            pet = self.is_available(ent["pet"])
            lab = self.is_available(ent["label"])
            if ct and pet and lab:
                out["complete"].append(c)
            elif ct and not pet:
                out["ct_only"].append(c)
            elif pet and not ct:
                out["pet_only"].append(c)
            elif lab and not (ct or pet):
                out["label_only"].append(c)
            else:
                out["none"].append(c)
        return out

    # -- extraction --------------------------------------------------------
    def extract(self, e: Entry, dest: Path, rename: str | None = None) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / (rename or Path(e.name).name)
        tmp = target.with_suffix(target.suffix + ".part")
        with self.zf.open(e.name) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        os.replace(tmp, target)
        return target


def open_catalog() -> Catalog:
    arch = PartialArchive(patches=load_patches())
    return Catalog(arch)


# --------------------------------------------------------------------- CLI --

def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024
    return ""


def cmd_layout(args) -> None:
    cat = open_catalog()
    groups: list[tuple[str, list[Entry]]] = []

    def add(label, pred):
        sel = [e for e in cat.entries if pred(e)]
        if sel:
            groups.append((label, sel))

    add("meta", lambda e: "/" not in e.name[len(TOPDIR) + 1:])
    add("psma PET (_0001)", lambda e: "/imagesTr/psma_" in e.name and e.name.endswith("_0001.nii.gz"))
    add("psma CT  (_0000)", lambda e: "/imagesTr/psma_" in e.name and e.name.endswith("_0000.nii.gz"))
    add("fdg  images", lambda e: "/imagesTr/fdg_" in e.name)
    add("labelsTr", lambda e: "/labelsTr/" in e.name)
    dl = cat.archive.downloaded
    print(f"archive size {ZIP_SIZE} ({human(ZIP_SIZE)}), downloaded "
          f"{dl} ({human(dl)}, {100 * dl / ZIP_SIZE:.2f}%)")
    for label, sel in groups:
        lo, hi = cat.region(sel)
        print(f"{label:>18}: {len(sel):5d} files  bytes {lo:>14,} .. {hi:>14,}  "
              f"({human(hi - lo)})  starts at {100 * lo / ZIP_SIZE:6.2f}%")


def cmd_meta(args) -> None:
    cat = open_catalog()
    out = Path(args.out)
    for e in cat.meta():
        p = cat.extract(e, out)
        print(f"  {p}  {human(p.stat().st_size)}")


def cmd_avail(args) -> None:
    cat = open_catalog()
    av = cat.available_cases()
    dl = cat.archive.downloaded
    summary = {
        "downloaded_bytes": dl,
        "downloaded_frac": dl / ZIP_SIZE,
        "counts": {k: len(v) for k, v in av.items()},
        "complete_fdg": len([c for c in av["complete"] if c.startswith("fdg_")]),
        "complete_psma": len([c for c in av["complete"] if c.startswith("psma_")]),
        "complete": av["complete"],
        "labels_available": sum(1 for e in cat.labels() if cat.is_available(e)),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "complete"}, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({**summary, "all": av}, indent=1))
        print(f"wrote {args.json}")


def cmd_labels(args) -> None:
    cat = open_catalog()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in cat.labels()
            if not (out / Path(e.name).name).exists() and cat.is_available(e)]
    print(f"{len(todo)} labels to extract "
          f"({len(cat.labels())} total, {len(list(out.glob('*.nii.gz')))} already on disk)")
    t0 = time.time()
    for i, e in enumerate(todo, 1):
        cat.extract(e, out)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done, {time.time() - t0:.0f}s")


def cmd_cases(args) -> None:
    """Extract N complete cases into an nnU-Net style images/ + labels/ folder."""
    cat = open_catalog()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.cases:
        pick = args.cases
    else:
        av = cat.available_cases()["complete"]
        pools = {"psma": [c for c in av if c.startswith("psma_")],
                 "fdg": [c for c in av if c.startswith("fdg_")]}
        # round-robin so we stay balanced, but top up from whichever tracer has
        # cases if the other is not downloaded yet
        pick, i = [], 0
        while len(pick) < args.n and any(len(v) > i for v in pools.values()):
            for v in pools.values():
                if i < len(v) and len(pick) < args.n:
                    pick.append(v[i])
            i += 1
    print(f"extracting {len(pick)} cases to {out}")
    for c in pick:
        ent = cat.case_entries(c)
        if not all(cat.is_available(v) for v in ent.values()):
            print(f"  SKIP (incomplete) {c}")
            continue
        t0 = time.time()
        cat.extract(ent["ct"], out / "images")
        cat.extract(ent["pet"], out / "images")
        cat.extract(ent["label"], out / "labels")
        sz = sum(ent[k].file_size for k in ent)
        print(f"  OK {c}  {human(sz)} in {time.time() - t0:.1f}s", flush=True)


def cmd_fetch_labels(args) -> None:
    """Range-fetch the whole labelsTr region (~0.26 GB) so labels are usable now."""
    cat = open_catalog()
    lo, hi = cat.region(cat.labels())
    if cat.archive.available(lo, hi - lo + 1):
        print("labels region already covered")
        return
    add_patch(lo, hi, "labelsTr")


def cmd_fetch_cases(args) -> None:
    """Range-fetch the missing members of specific cases (e.g. PSMA CT)."""
    cat = open_catalog()
    cases = args.cases
    if not cases:
        av = cat.available_cases()
        cases = (av["pet_only"] + av["ct_only"])[:args.n]
    for c in cases:
        ent = cat.case_entries(c)
        need = [(k, e) for k, e in ent.items() if e is not None and not cat.is_available(e)]
        for k, e in need:
            lo = e.header_offset
            hi = min(e.end_offset(slack=1024), ZIP_SIZE - 1)
            add_patch(lo, hi, f"{k}_{abs(hash(c)) % (10 ** 12)}")
    print("refresh the catalog to see the new availability")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("layout").set_defaults(fn=cmd_layout)

    p = sub.add_parser("meta")
    p.add_argument("--out", default="/content/drive/MyDrive/autoPET/meta")
    p.set_defaults(fn=cmd_meta)

    p = sub.add_parser("avail")
    p.add_argument("--json", default=None)
    p.set_defaults(fn=cmd_avail)

    p = sub.add_parser("labels")
    p.add_argument("--out", default="/content/drive/MyDrive/autoPET/labelsTr")
    p.set_defaults(fn=cmd_labels)

    p = sub.add_parser("cases")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default="/content/work/testcase")
    p.add_argument("--cases", nargs="*", default=None)
    p.set_defaults(fn=cmd_cases)

    sub.add_parser("fetch-labels").set_defaults(fn=cmd_fetch_labels)

    p = sub.add_parser("fetch-cases")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--cases", nargs="*", default=None)
    p.set_defaults(fn=cmd_fetch_cases)

    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
