"""Recreate the small MHD format fixture from its frozen parent without acquiring new data."""

from pathlib import Path

from echoprep.audit.identity import content_sha256, resolve_case_path


def materialize_derived(rows: list[dict], datasets: dict, root: Path) -> None:
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        if row.get("reader") != "sitk" or not row.get("derived_from"):
            continue
        target = resolve_case_path(row, datasets, root)
        if target.exists():
            continue
        parent = by_id[row["derived_from"]]
        source = resolve_case_path(parent, datasets, root)
        if content_sha256(source) != parent["content_sha256"]:
            raise ValueError("MHD parent content no longer matches the frozen identity")
        import shutil

        import SimpleITK as sitk

        target.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.ReadImage(str(source)), str(target))
        for cfg in source.parent.glob("Info_*.cfg"):
            shutil.copyfile(cfg, target.parent / cfg.name)
        if content_sha256(target) != row["content_sha256"]:
            raise ValueError(
                "Regenerated MHD differs from frozen identity; check SimpleITK version"
            )
