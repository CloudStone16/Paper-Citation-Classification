"""Download the Letchford et al. (2015) dataset and verify its checksum."""
import hashlib

import pandas as pd
import requests

from config import RAW_CSV, RAW_MD5, RAW_URL


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    if RAW_CSV.exists() and md5sum(RAW_CSV) == RAW_MD5:
        print(f"Already downloaded: {RAW_CSV}")
        return
    print(f"Downloading {RAW_URL}")
    with requests.get(RAW_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(RAW_CSV, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    digest = md5sum(RAW_CSV)
    if digest != RAW_MD5:
        raise ValueError(f"MD5 mismatch: expected {RAW_MD5}, got {digest}")
    print("MD5 verified")


if __name__ == "__main__":
    download()
    df = pd.read_csv(RAW_CSV)
    print(f"\nRows: {len(df):,}  Columns: {list(df.columns)}")
    print(df.dtypes.to_string())
    print("\nSample:")
    print(df.head().to_string())
