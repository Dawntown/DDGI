#!/usr/bin/env python3
"""Download and preprocess the small CellNavi T cell tutorial data for DDGI.

The script expects two raw AnnData files from the CellNavi tutorial dataset:

- Re-stimulated_t_example_train.h5ad
- Resting_t_example_test.h5ad

By default it tries the public Dropbox folder used by CellNavi/DDGI. If the
folder link changes, download the two files manually and rerun this script with
--raw-dir pointing to their location.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import scanpy as sc

RAW_FILES = {
    "train": "Re-stimulated_t_example_train.h5ad",
    "test": "Resting_t_example_test.h5ad",
}
DROPBOX_FOLDER_URL = (
    "https://www.dropbox.com/scl/fo/rq9klah7vqksn6e66dsae/"
    "AK3DJ2sxwL3MoWCOcQ9ZfFE?rlkey=1t4kz2vraif0ifu72c6gmo6xl&dl=0"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _download_with_curl(url: str, output: Path) -> bool:
    tmp_output = output.with_suffix(output.suffix + ".part")
    if tmp_output.exists():
        tmp_output.unlink()
    try:
        _run(["curl", "-L", "--fail", "-o", str(tmp_output), url], cwd=output.parent)
        if tmp_output.exists() and tmp_output.stat().st_size > 0:
            tmp_output.replace(output)
            return True
        return False
    except (FileNotFoundError, subprocess.CalledProcessError):
        if tmp_output.exists():
            tmp_output.unlink()
        return False


def _dropbox_file_url(folder_url: str, file_name: str) -> str:
    base_url, _, query = folder_url.partition("?")
    params = {}
    for item in query.split("&"):
        if not item:
            continue
        key, _, value = item.partition("=")
        params[key] = value
    params["preview"] = file_name
    params["dl"] = "1"
    return base_url + "?" + urlencode(params)


def _download_raw_files(raw_dir: Path, folder_url: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in RAW_FILES.values() if not (raw_dir / name).exists()]
    if not missing:
        return

    print("Raw tutorial files are missing:", ", ".join(missing))
    print("Attempting to download only the required .h5ad files from Dropbox.")

    for file_name in missing:
        target = raw_dir / file_name
        url = _dropbox_file_url(folder_url, file_name)
        if _download_with_curl(url, target):
            print(f"Downloaded {target}")

    still_missing = [name for name in RAW_FILES.values() if not (raw_dir / name).exists()]
    if still_missing:
        raise FileNotFoundError(
            "Could not download all raw tutorial files automatically. "
            f"Download {', '.join(still_missing)} from {folder_url} and place them in {raw_dir}."
        )


def preprocess(raw_dir: Path, out_dir: Path, n_top_genes: int, seed: int) -> None:
    train = sc.read_h5ad(raw_dir / RAW_FILES["train"])
    test = sc.read_h5ad(raw_dir / RAW_FILES["test"])
    adata = sc.concat({"train": train, "test": test}, label="split")
    adata.obs_names_make_unique()
    adata.obs["perturbation"] = adata.obs["perturbation"].replace({"NO-TARGET": "control"})
    adata.uns["organism"] = adata.uns.get("organism", "human")

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")

    out_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_dir / "preprocessed.h5ad", compression="gzip")

    rng = np.random.default_rng(seed)
    split_df = (
        adata.obs[["split", "condition"]]
        .reset_index(names="cell")
        .rename(columns={"condition": "subsplit"})
    )
    split_df["presplit"] = rng.choice(
        ["train", "val", "test"],
        size=split_df.shape[0],
        p=[0.7, 0.1, 0.2],
    )

    train_on_stim = split_df.copy()
    train_on_stim["split"] = train_on_stim[["subsplit", "presplit"]].apply(
        lambda row: row["presplit"] if row["subsplit"] == "Re-stimulated" else "test",
        axis=1,
    )
    train_on_stim[["cell", "split", "subsplit"]].to_csv(
        out_dir / "split_trainonstimulated.csv", index=False
    )

    train_on_rest = split_df.copy()
    train_on_rest["split"] = train_on_rest[["subsplit", "presplit"]].apply(
        lambda row: row["presplit"] if row["subsplit"] == "Resting" else "test",
        axis=1,
    )
    train_on_rest[["cell", "split", "subsplit"]].to_csv(
        out_dir / "split_trainonresting.csv", index=False
    )

    print("Wrote:")
    for path in [
        out_dir / "preprocessed.h5ad",
        out_dir / "split_trainonstimulated.csv",
        out_dir / "split_trainonresting.csv",
    ]:
        print(f"  {path}")


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "datasets" / "schmidt_tutorial" / "raw")
    parser.add_argument("--out-dir", type=Path, default=root / "datasets" / "schmidt_tutorial")
    parser.add_argument("--dropbox-url", default=DROPBOX_FOLDER_URL)
    parser.add_argument("--n-top-genes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        _download_raw_files(args.raw_dir, args.dropbox_url)
    else:
        missing = [name for name in RAW_FILES.values() if not (args.raw_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing raw files in {args.raw_dir}: {', '.join(missing)}")

    preprocess(args.raw_dir, args.out_dir, args.n_top_genes, args.seed)


if __name__ == "__main__":
    main()
