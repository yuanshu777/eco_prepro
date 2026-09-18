"""Small-sample acquisition. Local-first, bounded public downloads, no archive downloads."""

from __future__ import annotations

import hashlib
import os
from itertools import islice
from pathlib import Path
from urllib.request import urlopen

ACQUISITION_STATUSES = {
    "FOUND_LOCAL",
    "DOWNLOADED",
    "AUTH_REQUIRED",
    "ARCHIVE_ONLY",
    "UNAVAILABLE",
    "SKIPPED",
}


def acquire_samples(
    name: str,
    spec: dict,
    project_root: Path,
    *,
    download: bool = True,
    limit: int = 3,
    opener=urlopen,
) -> dict:
    if not 1 <= limit <= 10:
        raise ValueError("sample limit must be between 1 and 10")
    root = project_root / spec["root"]
    result = {
        k: spec.get(k)
        for k in (
            "source_url",
            "access_method",
            "license",
            "redistributable",
            "redistribution_restrictions",
        )
    }
    result.update(dataset=name, paths=[], status="UNAVAILABLE", reason="", sample_only=True)
    files = (
        list(islice((p for p in root.glob(spec.get("videos_glob", "**/*")) if p.is_file()), limit))
        if root.exists()
        else []
    )
    if files:
        return dict(
            result,
            status="FOUND_LOCAL",
            paths=[str(p) for p in files],
            reason="Existing local samples reused",
        )
    if not download:
        return dict(result, status="SKIPPED", reason="Local-only mode; no matching local samples")
    access = spec.get("access_method")
    if access == "restricted":
        required = spec.get("credential_env", [])
        present = bool(required) and all(os.environ.get(k) for k in required)
        return dict(
            result,
            status="SKIPPED" if present else "AUTH_REQUIRED",
            reason="Credential variables present; no approved per-file acquisition adapter configured"
            if present
            else "Authorized access required; place approved samples under the configured root",
        )
    if access == "archive":
        return dict(
            result,
            status="ARCHIVE_ONLY",
            reason="No supported individual-sample endpoint; full archive downloads are disabled",
        )
    if access != "public_samples" or not spec.get("sample_urls"):
        return dict(result, status="UNAVAILABLE", reason="No sample endpoint configured")
    errors = []
    for sample in spec["sample_urls"][:limit]:
        dest = root / sample["filename"]
        if not dest.resolve().is_relative_to(root.resolve()):
            raise ValueError("sample filename escapes dataset root")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        try:
            cap = min(int(sample.get("max_bytes", 16 * 1024 * 1024)), 32 * 1024 * 1024)
            with opener(sample["url"], timeout=30) as response, tmp.open("wb") as out:
                count, digest = 0, hashlib.sha256()
                while block := response.read(65536):
                    count += len(block)
                    if count > cap:
                        raise ValueError("sample exceeds download byte limit")
                    out.write(block)
                    digest.update(block)
            if not count:
                raise ValueError("empty sample response")
            if sample.get("sha256") and digest.hexdigest() != sample["sha256"]:
                raise ValueError("sample checksum mismatch")
            if spec.get("reader") == "dicom":
                import pydicom

                ds = pydicom.dcmread(tmp, stop_before_pixels=True)
                if "Rows" not in ds or "Columns" not in ds:
                    raise ValueError("sample is not a DICOM image")
            tmp.replace(dest)
            result["paths"].append(str(dest))
        except Exception as exc:  # noqa: BLE001 - report failures and continue other cases  # independent sources must not break the acquisition report
            errors.append(f"{sample['filename']}: {type(exc).__name__}: {exc}")
        finally:
            tmp.unlink(missing_ok=True)
    result["status"] = "DOWNLOADED" if result["paths"] else "UNAVAILABLE"
    result["reason"] = "; ".join(errors) if errors else "Bounded public samples acquired"
    return result
