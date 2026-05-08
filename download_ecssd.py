from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


ECSSD_URLS = {
    "images.zip": [
        "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip",
        "http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip",
    ],
    "ground_truth_mask.zip": [
        "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip",
        "http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip",
    ],
}


def _valid_zip(path: Path) -> bool:
    return path.exists() and zipfile.is_zipfile(path)


def download_file(urls: list[str], destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _valid_zip(destination):
        print(f"Already downloaded: {destination}")
        return
    if destination.exists():
        print(f"Removing incomplete or invalid download: {destination}")
        destination.unlink()

    part_path = destination.with_suffix(destination.suffix + ".part")
    part_path.unlink(missing_ok=True)
    errors: list[str] = []
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                print(f"Downloading {url} (attempt {attempt}/{retries})")
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=90) as response, part_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
                part_path.replace(destination)
                if not _valid_zip(destination):
                    raise zipfile.BadZipFile(f"Downloaded file is not a valid zip: {destination}")
                print(f"Saved {destination}")
                return
            except Exception as exc:
                errors.append(f"{url} attempt {attempt}: {exc}")
                part_path.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                print(f"Download failed: {exc}")

    raise RuntimeError("Could not download ECSSD file.\n" + "\n".join(errors[-6:]))


def extract_zip(zip_path: Path, destination: Path) -> None:
    marker = destination / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"Already extracted: {zip_path.name}")
        return
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the ECSSD saliency dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ecssd"))
    parser.add_argument("--keep-zips", action="store_true", help="Keep downloaded zip files after extraction.")
    args = parser.parse_args()

    zip_dir = args.data_dir / "_zips"
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for filename, urls in ECSSD_URLS.items():
        zip_path = zip_dir / filename
        download_file(urls, zip_path)
        extract_zip(zip_path, args.data_dir)
        if not args.keep_zips:
            zip_path.unlink(missing_ok=True)

    print("ECSSD ready.")
    print(f"Data directory: {args.data_dir.resolve()}")
    print("Next: python train.py --data-dir data/ecssd --output-dir checkpoints/baseline")


if __name__ == "__main__":
    main()
