"""Download only the daily CSV, verify its bytes, and write it atomically."""

from io import BytesIO
import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from bike_backtest.core import DATA_SHA256, DATA_URL


def main():
    with urlopen(DATA_URL, timeout=60) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as zipped:
        daily = zipped.read("day.csv")
    if hashlib.sha256(daily).hexdigest() != DATA_SHA256:
        raise ValueError(
            "UCI data changed: inspect before updating the pinned checksum."
        )
    path = Path("data/day.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(daily)
    temporary.replace(path)
    print(f"Verified and saved {path}")


if __name__ == "__main__":
    main()
