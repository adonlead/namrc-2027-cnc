
"""
NAMRC 2027 — NASA Milling Data Pipeline
Version 0.3.1

Automatically downloads and inspects NASA Milling data.

Supports:
- ZIP archives
- Nested ZIP archives
- MATLAB .mat files (SciPy-compatible versions)
- Diagnostic JSON output

No synthetic experimental data or predictions are generated.
"""

import io
import json
import hashlib
import zipfile
import datetime
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat


NASA_URL = (
    "https://phm-datasets.s3.amazonaws.com/"
    "NASA/3.+Milling.zip"
)

OUTPUT_DIR = Path("public/api")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DOWNLOAD = 400 * 1024 * 1024
MAX_FILE = 500 * 1024 * 1024
MAX_NESTING = 3
MAX_ENTRIES = 5000

inventory = []
mat_inventory = []
diagnostics = []
records = []


def timestamp():
    return datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()


def save_json(filename, data):
    path = OUTPUT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        )

    print("Saved:", path)


def download_archive():
    print("Downloading NASA Milling archive...")

    request = urllib.request.Request(
        NASA_URL,
        headers={
            "User-Agent": "NAMRC-2027-Research/0.3.1"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=120
    ) as response:
        raw = response.read(MAX_DOWNLOAD + 1)

    if len(raw) > MAX_DOWNLOAD:
        raise RuntimeError(
            "NASA archive exceeds download limit"
        )

    print("Download completed.")
    print("Archive size:", len(raw), "bytes")

    return raw


def normalize(value):
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return [
                normalize(x)
                for x in value.reshape(-1)
            ]

        return normalize(value.tolist())

    if isinstance(value, np.void):
        return {
            k: normalize(value[k])
            for k in value.dtype.names
        }

    if hasattr(value, "_fieldnames"):
        return {
            k: normalize(getattr(value, k))
            for k in value._fieldnames
        }

    if isinstance(value, np.generic):
        return normalize(value.item())

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace"
        )

    if isinstance(value, dict):
        return {
            str(k): normalize(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize(x) for x in value]

    return value


def numeric_scalar(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]

    if isinstance(value, (int, float)):
        if np.isfinite(value):
            return float(value)

    return None


def summarize_record(record, index, source):
    result = {
        "experiment_index": index,
        "source_file": source
    }

    for key, value in record.items():
        scalar = numeric_scalar(value)

        if scalar is not None:
            result[key] = scalar
            continue

        if isinstance(value, str):
            result[key] = value
            continue

        if isinstance(value, list):
            try:
                array = np.asarray(value)

                if (
                    array.size == 0
                    or array.size > 2000000
                    or not np.issubdtype(
                        array.dtype,
                        np.number
                    )
                ):
                    continue

                values = array.astype(float).ravel()
                values = values[np.isfinite(values)]

                if values.size == 0:
                    continue

                result[key + "_mean"] = float(
                    np.mean(values)
                )

                result[key + "_std"] = float(
                    np.std(values)
                )

                result[key + "_rms"] = float(
                    np.sqrt(np.mean(values ** 2))
                )

                result[key + "_samples"] = int(
                    values.size
                )

            except (ValueError, TypeError):
                continue

    return result


def inspect_mat(data, filename):
    print("Inspecting MATLAB file:", filename)

    try:
        obj = loadmat(
            io.BytesIO(data),
            simplify_cells=True
        )

        variables = [
            key for key in obj.keys()
            if not key.startswith("__")
        ]

        mat_inventory.append({
            "file": filename,
            "variables": variables
        })

        print("MATLAB variables:", variables)

        for key in variables:
            value = normalize(obj[key])

            candidates = (
                value if isinstance(value, list)
                else [value]
            )

            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue

                result = summarize_record(
                    candidate,
                    len(records) + 1,
                    filename
                )

                result["source_variable"] = key

                if len(result) > 3:
                    records.append(result)

    except Exception as exc:
        message = (
            filename + ": " +
            type(exc).__name__ + ": " +
            str(exc)
        )

        diagnostics.append(message)
        print("MATLAB processing warning:", message)


def inspect_zip(raw, label="NASA archive", depth=0):
    if depth > MAX_NESTING:
        diagnostics.append(
            "Maximum nesting reached: " + label
        )
        return

    print()
    print("Inspecting ZIP:", label)

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()

            print("Total entries:", len(entries))

            if len(entries) > MAX_ENTRIES:
                diagnostics.append(
                    "Too many archive entries: " + label
                )
                return

            for entry in entries:
                name = entry.filename
                lower = name.lower()

                print(
                    "FILE:",
                    name,
                    "| SIZE:",
                    entry.file_size
                )

                inventory.append({
                    "archive": label,
                    "name": name,
                    "size": entry.file_size,
                    "depth": depth
                })

                if entry.is_dir():
                    continue

                if entry.file_size > MAX_FILE:
                    diagnostics.append(
                        "File exceeds size limit: " + name
                    )
                    continue

                if lower.endswith(".zip"):
                    print("Nested ZIP detected:", name)

                    nested = archive.read(entry)

                    inspect_zip(
                        nested,
                        label + "/" + name,
                        depth + 1
                    )

                elif lower.endswith(".mat"):
                    print("MATLAB file detected:", name)

                    inspect_mat(
                        archive.read(entry),
                        label + "/" + name
                    )

                elif lower.endswith((
                    ".csv", ".txt", ".dat",
                    ".xlsx", ".json", ".h5", ".hdf5"
                )):
                    print(
                        "Potential dataset file:",
                        name
                    )

    except Exception as exc:
        message = (
            label + ": " +
            type(exc).__name__ + ": " +
            str(exc)
        )

        diagnostics.append(message)
        print("Archive inspection error:", message)


def main():
    print("=" * 60)
    print("NAMRC 2027 — NASA Milling Data Pipeline")
    print("Version 0.3.1")
    print("=" * 60)

    raw = download_archive()

    sha256 = hashlib.sha256(raw).hexdigest()

    print("SHA256:", sha256)

    inspect_zip(raw)

    print()
    print("=" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 60)

    print("Archive entries:", len(inventory))
    print("MATLAB files:", len(mat_inventory))
    print("Parsed records:", len(records))
    print("Warnings:", len(diagnostics))

    if records:
        status = "parsed_records"
    elif mat_inventory:
        status = "matlab_found_no_records"
    else:
        status = "inspection_only"

    manifest = {
        "dataset": "NASA Milling Wear",
        "source_url": NASA_URL,
        "retrieved_utc": timestamp(),
        "sha256": sha256,
        "archive_bytes": len(raw),
        "status": status,
        "archive_files": inventory,
        "mat_inventory": mat_inventory,
        "record_count": len(records),
        "warnings": diagnostics,
        "research_notice": (
            "Historical NASA milling experiments. "
            "No predictive model has been validated. "
            "Parsed records require verification "
            "against original dataset documentation."
        )
    }

    experiments = {
        "metadata": {
            "dataset": "NASA Milling Wear",
            "sha256": sha256,
            "generated_utc": timestamp(),
            "record_count": len(records),
            "validated": False
        },
        "records": records
    }

    save_json("manifest.json", manifest)
    save_json("experiments.json", experiments)

    print()
    print("NASA dataset inspection completed.")

    if not records:
        print(
            "No experimental records extracted yet. "
            "Review the archive inventory to "
            "identify the correct data format."
        )
    else:
        print(
            "Experimental records extracted:",
            len(records)
        )

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
