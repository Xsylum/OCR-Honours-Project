"""Download the Catalogue of Life ColDP archive (contains NameUsage.tsv).

Usage:
    python download_col.py                  # the pinned release (dataset 316595)
    python download_col.py --date 2025-08-20  # a different monthly release instead

Can also be imported:
    from download_col import ensure_col_zip
    zip_path = ensure_col_zip()
"""
import argparse
import os
import sys
import urllib.request

# Pinned to the exact release used to build this dataset (ChecklistBank dataset 316595).
PINNED_URL = "https://api.checklistbank.org/dataset/316595/export.zip?extended=true&format=ColDP"
BASE = "https://download.checklistbank.org/col"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ZIP = os.path.join(HERE, "col_coldp.zip")


def build_url(date=None):
    if date:
        return f"{BASE}/monthly/{date}_coldp.zip"
    return PINNED_URL


def download(url, dest):
    tmp = dest + ".part"  # download to a temp name so a failed run never leaves a corrupt zip
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
            else:
                sys.stdout.write(f"\r  {done / 1e6:,.0f} MB")
            sys.stdout.flush()
    os.replace(tmp, dest)
    print(f"\nSaved to {dest}")


def ensure_col_zip(dest=DEFAULT_ZIP, date=None):
    """Return the path to the zip, downloading it first if it isn't there."""
    if os.path.exists(dest):
        return dest
    download(build_url(date), dest)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="monthly release date, e.g. 2025-08-20")
    parser.add_argument("--force", action="store_true", help="re-download even if the zip exists")
    args = parser.parse_args()

    if args.force and os.path.exists(DEFAULT_ZIP):
        os.remove(DEFAULT_ZIP)
    ensure_col_zip(date=args.date)