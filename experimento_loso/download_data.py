"""Download and verify the NinaPro DB3 raw files used by the LOSO experiment.

The files are fetched from a GitHub Release and checked against the SHA-256
manifest in ``data/raw_manifest.sha256``. Only the Python standard library is
used, so the script can run before any dependency is installed.

Data source: NinaPro (https://ninapro.hevs.ch/). Cite Atzori et al. (2014),
Scientific Data 1:140053, doi 10.1038/sdata.2014.53.

Usage (from the project root)::

    python experimento_loso/download_data.py
    python experimento_loso/download_data.py --subjects 1 2 --retries 5
    python experimento_loso/download_data.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DEST: Path = ROOT / "data" / "raw"
MANIFEST_PATH: Path = ROOT / "data" / "raw_manifest.sha256"
DEFAULT_BASE_URL: str = (
    "https://github.com/jhonosorno-cmd/clasificador-semg/releases/download/data-v1/"
)
CHUNK_SIZE: int = 1024 * 1024
PROGRESS_INTERVAL_S: float = 1.0
BACKOFF_BASE_S: float = 2.0
HTTP_TIMEOUT_S: float = 60.0
SHA256_HEX_LENGTH: int = 64
NINAPRO_NOTICE: str = (
    "Data: NinaPro DB3 (https://ninapro.hevs.ch/). Please cite: Atzori et al. (2014), "
    "Scientific Data 1:140053, doi 10.1038/sdata.2014.53."
)


def load_manifest(path: Path) -> dict[str, str]:
    """Parse the SHA-256 manifest into a filename-to-digest mapping.

    Args:
        path: Manifest file with lines of the form ``<sha256hex>  <filename>``.

    Returns:
        Mapping from filename to lowercase SHA-256 hex digest, in manifest order.

    Raises:
        OSError: If the manifest cannot be read.
        ValueError: If a non-empty line is malformed.
    """
    entries: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != SHA256_HEX_LENGTH:
            raise ValueError(f"{path}:{line_no}: malformed manifest line")
        entries[parts[1].strip()] = parts[0].lower()
    return entries


def select_names(manifest: dict[str, str], subjects: list[int] | None) -> list[str]:
    """Select manifest filenames for the requested subjects.

    Args:
        manifest: Filename-to-digest mapping from :func:`load_manifest`.
        subjects: Subject numbers to keep, or ``None`` for every file.

    Returns:
        Filenames in manifest order, without duplicates.

    Raises:
        ValueError: If a requested subject has no file in the manifest.
    """
    if subjects is None:
        return list(manifest)
    selected: list[str] = []
    for subject in subjects:
        prefix = f"S{subject}_"
        matches = [name for name in manifest if name.startswith(prefix)]
        if not matches:
            raise ValueError(f"subject {subject} has no file in the manifest")
        for name in matches:
            if name not in selected:
                selected.append(name)
    return selected


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, reading it in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase SHA-256 hex digest.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _print_progress(done: int, total: int | None) -> None:
    """Redraw the in-place progress line in megabytes.

    Args:
        done: Bytes downloaded so far.
        total: Expected size in bytes from Content-Length, or ``None`` if unknown.
    """
    done_mb = done / (1024 * 1024)
    if total:
        total_mb = total / (1024 * 1024)
        line = f"\r    {done_mb:8.1f} / {total_mb:.1f} MB"
    else:
        line = f"\r    {done_mb:8.1f} MB"
    sys.stdout.write(line)
    sys.stdout.flush()


def download_to_part(url: str, part: Path) -> None:
    """Stream a URL into a partial file, reporting progress about once per second.

    Args:
        url: Absolute URL of the file.
        part: Destination ``.part`` path; overwritten if it exists.

    Raises:
        urllib.error.URLError: On HTTP or network failure.
        OSError: On a local write failure or timeout.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "clasificador-semg-download"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            length_header = response.headers.get("Content-Length")
            total = int(length_header) if length_header and length_header.isdigit() else None
            done = 0
            last_print = 0.0
            with part.open("wb") as handle:
                while chunk := response.read(CHUNK_SIZE):
                    handle.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= PROGRESS_INTERVAL_S:
                        _print_progress(done, total)
                        last_print = now
            _print_progress(done, total)
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()


def fetch_verified(name: str, expected: str, dest: Path, base_url: str, retries: int) -> str | None:
    """Download one file to ``.part``, verify it, and atomically move it into place.

    The final file is replaced only after the new download matches the manifest
    digest. Up to ``retries`` extra attempts are made after the first, with a
    linear backoff between attempts.

    Args:
        name: Filename to fetch.
        expected: Expected lowercase SHA-256 hex digest.
        dest: Destination directory.
        base_url: Base URL ending in a slash (or without; it is normalized).
        retries: Number of retries after the first attempt.

    Returns:
        ``None`` on success, otherwise a description of the last failure.
    """
    final = dest / name
    part = dest / f"{name}.part"
    url = f"{base_url.rstrip('/')}/{name}"
    last_error = "unknown error"
    for attempt in range(retries + 1):
        if attempt:
            wait = BACKOFF_BASE_S * attempt
            print(f"    retry {attempt}/{retries} in {wait:.0f}s after: {last_error}")
            time.sleep(wait)
        try:
            download_to_part(url, part)
        except (urllib.error.URLError, OSError) as exc:
            part.unlink(missing_ok=True)
            last_error = f"download failed: {exc}"
            continue
        actual = sha256_of(part)
        if actual == expected:
            part.replace(final)
            return None
        part.unlink(missing_ok=True)
        last_error = f"hash mismatch (expected {expected[:12]}..., got {actual[:12]}...)"
    return last_error


def process_one(
    name: str,
    expected: str,
    dest: Path,
    base_url: str,
    retries: int,
    verify_only: bool,
) -> bool:
    """Check or fetch a single file and report its status.

    Args:
        name: Filename to process.
        expected: Expected lowercase SHA-256 hex digest.
        dest: Destination directory.
        base_url: Base URL for downloads.
        retries: Number of retries after the first download attempt.
        verify_only: If ``True``, only check presence and hashes; never download.

    Returns:
        ``True`` if the file is present and verified, otherwise ``False``.

    Raises:
        OSError: If the existing file cannot be read.
    """
    final = dest / name
    if final.is_file():
        if sha256_of(final) == expected:
            print(f"{name}: OK (already present)")
            return True
        print(
            f"WARNING {name}: existing file does not match the manifest hash; "
            "it is kept until a verified replacement is downloaded."
        )
        if verify_only:
            print(f"{name}: BAD (hash mismatch)")
            return False
    elif verify_only:
        print(f"{name}: MISSING")
        return False

    print(f"{name}: downloading")
    error = fetch_verified(name, expected, dest, base_url, retries)
    if error is None:
        print(f"{name}: OK (downloaded and verified)")
        return True
    print(f"{name}: FAILED: {error}")
    return False


def main(argv: list[str] | None = None) -> int:
    """Run the download-and-verify workflow from the command line.

    Args:
        argv: Argument list without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code: 0 if every requested file is OK, 1 if any failed,
        2 on configuration errors.
    """
    parser = argparse.ArgumentParser(
        description="Download and verify NinaPro DB3 raw files from a GitHub Release."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Subject numbers to fetch (default: every file in the manifest).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL that the filenames are appended to.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries after the first failed attempt per file (default: 3).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check presence and hashes; do not download.",
    )
    args = parser.parse_args(argv)
    if args.retries < 0:
        parser.error("--retries must be 0 or greater")

    print(NINAPRO_NOTICE)

    try:
        manifest = load_manifest(MANIFEST_PATH)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read manifest: {exc}", file=sys.stderr)
        return 2

    try:
        names = select_names(manifest, args.subjects)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    dest: Path = args.dest
    if not args.verify_only:
        dest.mkdir(parents=True, exist_ok=True)

    ok_names: list[str] = []
    failed_names: list[str] = []
    for name in names:
        if process_one(name, manifest[name], dest, args.base_url, args.retries, args.verify_only):
            ok_names.append(name)
        else:
            failed_names.append(name)

    print()
    print(f"Summary: {len(ok_names)} OK, {len(failed_names)} failed")
    if failed_names:
        print("Failed: " + ", ".join(failed_names))
    return 0 if not failed_names else 1


if __name__ == "__main__":
    sys.exit(main())
