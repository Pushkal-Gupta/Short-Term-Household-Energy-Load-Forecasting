"""Download the Low Carbon London smart-meter dataset from the London Datastore.

    python -m src.data.download                # ~800 MB zip into data/raw/
    python -m src.data.download --dest data/raw

Source: UK Power Networks, "SmartMeter Energy Consumption Data in London Households"
https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d
Check the licence on that page before redistributing. The zip does NOT need to be extracted:
the preprocessing step streams the CSVs straight out of it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import resolve

LCL_PAGE = "https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d"
LCL_PARTITIONED_URL = ("https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/"
                       "Partitioned%20LCL%20Data.zip")
LCL_PARTITIONED_BYTES = 795_722_689
LCL_ZIP_NAME = "Partitioned_LCL_Data.zip"


def download(dest: Path, retries: int = 10) -> Path:
    """Resumable download via curl (present on macOS, Linux and Windows 10+)."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / LCL_ZIP_NAME
    for attempt in range(1, retries + 1):
        if target.exists() and target.stat().st_size == LCL_PARTITIONED_BYTES:
            print(f"OK: {target} ({LCL_PARTITIONED_BYTES:,} bytes)")
            return target
        print(f"Downloading (attempt {attempt}) -> {target}")
        subprocess.run(["curl", "-L", "--http1.1", "-C", "-", "-o", str(target), LCL_PARTITIONED_URL],
                       check=False)
    size = target.stat().st_size if target.exists() else 0
    raise RuntimeError(f"Download incomplete ({size:,} of {LCL_PARTITIONED_BYTES:,} bytes). "
                       f"Download manually from {LCL_PAGE} and place the zip in {dest}.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="data/raw")
    args = ap.parse_args(argv)
    try:
        download(resolve(args.dest))
    except RuntimeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
