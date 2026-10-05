"""Analyze a frozen three-source EDA inventory; all row-level outputs stay local."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from matplotlib.ticker import PercentFormatter
from PIL import Image

from echoprep.cine import Cine
from echoprep.sector.layouts import ev9v_hint_mask
from echoprep.sector.motion import detect_sector
from echoprep.standardize import standardize_cine

SOURCES = ["EV9V", "MIMIC-IV-Echo", "EchoNet-Dynamic"]
COLORS = ["#147d80", "#c88722", "#4769a5"]
EP_VIEWS = [
    "A2C",
    "A3C",
    "A4C",
    "A5C",
    "Apical Doppler",
    "Doppler PLAX",
    "Doppler PSAX",
    "PLAX",
    "PSAX",
    "SSN",
    "Subcostal",
]
FAMILY5 = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
SEED = 20261005


def frames_for(row, indices):
    """Read original-order frames; preserve every requested index exactly."""
    indices = sorted({int(i) for i in indices})
    if row["dataset"] == "EV9V":
        return np.stack(
            [
                np.asarray(Image.open(Path(row["frames_dir"]) / f"{i}.jpg").convert("RGB"))
                for i in indices
            ]
        )
    if row["format"] == "DICOM":
        ims = []
        for arr in pydicom.pixels.iter_pixels(row["path"], indices=indices):
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.dtype != np.uint8:
                bits = row.get("bits_stored")
                hi = 2 ** int(bits) - 1 if pd.notna(bits) else max(float(arr.max()), 1)
                arr = (np.clip(arr.astype(float) / hi, 0, 1) * 255).astype(np.uint8)
            if row.get("photometric") == "MONOCHROME1":
                arr = 255 - arr
            ims.append(arr)
        if len(ims) != len(indices):
            raise ValueError("Decoded frame count differs from requested indices")
        return np.stack(ims)
    wanted = set(indices)
    ims = []
    with av.open(row["path"]) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                ims.append(frame.to_ndarray(format="rgb24"))
            if i >= indices[-1]:
                break
    if len(ims) != len(indices):
        raise ValueError("Missing requested video frame")
    return np.stack(ims)


def choose_qc(df):
    """Purposeful equal allocation across strata; never a prevalence sample."""
    rng = np.random.default_rng(SEED)
    chunks = []
    ev = df[df.dataset == "EV9V"].copy()
    en = df[df.dataset == "EchoNet-Dynamic"].copy()
    for frame, key, each in [(ev, "original_view", 2), (en, "split", 4)]:
        frame["length_bin"] = pd.cut(
            frame.duration_s, [0, 2, 5, np.inf], right=False, labels=["<2s", "2-5s", ">=5s"]
        )
        for _, group in frame.groupby([key, "length_bin"], observed=True):
            idx = rng.choice(group.index, min(each, len(group)), replace=False)
            chunks.append(frame.loc[idx].assign(sampling_role="stratified visual QC"))
    mi = df[
        (df.dataset == "MIMIC-IV-Echo") & (df.bmode_cine == True) & (df.study_complete == True)
    ].copy()
    subjects = sorted(mi.subject_id.dropna().unique())
    for subj in rng.choice(subjects, min(60, len(subjects)), replace=False):
        group = mi[mi.subject_id == subj]
        idx = rng.choice(group.index)
        chunks.append(
            group.loc[[idx]].assign(
                length_bin="", sampling_role="one B-mode loop per sampled patient"
            )
        )
    return pd.concat(chunks).sort_values(["dataset", "case_id"])


def qc_sample(df, out):
    selection = choose_qc(df)
    selection.to_csv(out / "tables/qc_sample_selection.csv", index=False)
    rows, frame_rows = [], []
    for i, r in enumerate(selection.to_dict("records"), 1):
        idx = np.unique(
            np.linspace(0, int(r["n_frames"]) - 1, min(16, int(r["n_frames"]))).round().astype(int)
        )
        result = {
            "case_id": r["case_id"],
            "dataset": r["dataset"],
            "n_requested": len(idx),
            "decode_error": "",
            "sampling_role": r["sampling_role"],
        }
        try:
            ims = frames_for(r, idx)
            gray = np.stack(
                [
                    cv2.cvtColor(
                        cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY
                    )
                    for a in ims
                ]
            )
            brightness = gray.mean(axis=(1, 2))
            sharpness = [cv2.Laplacian(a, cv2.CV_64F).var() for a in gray]
            result.update(
                brightness=float(np.median(brightness)),
                nonblack_fraction=float((gray > 6).mean()),
                temporal_std=float(gray.astype(float).std(axis=0).mean()),
                laplacian_variance=float(np.median(sharpness)),
                n_decoded=len(ims),
            )
            for j, k in enumerate(idx):
                frame_rows.append(
                    {
                        "case_id": r["case_id"],
                        "dataset": r["dataset"],
                        "frame_index": int(k),
                        "timestamp_s": float(k / r["fps"]) if pd.notna(r["fps"]) else None,
                        "brightness": float(brightness[j]),
                        "laplacian_variance": float(sharpness[j]),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - isolate and record per-file decode failures
            result["decode_error"] = type(exc).__name__ + ": " + str(exc)[:100]
        rows.append(result)
        if i % 25 == 0:
            print("QC frames", i, "/", len(selection), flush=True)
    pd.DataFrame(rows).to_csv(out / "tables/qc_sample_results.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(out / "tables/frame_qc.csv", index=False)
    return pd.DataFrame(rows)


def predictions(df, routing, out):
    rows = []
    folder = routing / "runs/_demo/realvideo/mimic"
    for r in df[df.dataset == "MIMIC-IV-Echo"].to_dict("records"):
        stem = Path(r["path"]).stem
        p = folder / f"{r['study_id']}_{stem}.npz"
        if not p.is_file():
            continue
        with np.load(p) as z:
            if int(z["n_frames"]) != int(r["n_frames"]):
                continue
            probs = z["raw_ep"]
            mean = probs.mean(axis=0)
            rows.append(
                {
                    "case_id": r["case_id"],
                    "dataset": r["dataset"],
                    "predicted_view": EP_VIEWS[int(mean.argmax())],
                    "mean_top_probability": float(mean.max()),
                    "n_sampled_frames": len(probs),
                    "frame_count": int(z["n_frames"]),
                    "source": "EchoPrime released classifier; pre-existing cache",
                    "aggregation": "argmax of mean raw 11-class probability",
                    "expert_label": False,
                }
            )
    pred = pd.DataFrame(rows)
    pred.to_csv(out / "tables/mimic_predicted_views.csv", index=False)
    return pred


def savefig(fig, out, name, caption, evidence, registry):
    fig.savefig(out / "figures" / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "figures" / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    registry.append({"name": name, "caption": caption, "evidence": evidence})


def summaries(df, pred, out):
    rows = []
    for source in SOURCES:
        g = df[df.dataset == source]
        timed = g[(g.dynamic == True) & g.duration_s.notna()]
        rows.append(
            {
                "dataset": source,
                "files": len(g),
                "frames": int(g.n_frames.sum()),
                "dynamic": int((g.dynamic == True).sum()),
                "header_errors": int(g.header_error.notna().sum()),
                "total_hours": float(timed.duration_s.sum() / 3600),
                "duration_median": float(timed.duration_s.median()),
                "duration_p05": float(timed.duration_s.quantile(0.05)),
                "duration_p95": float(timed.duration_s.quantile(0.95)),
                "duration_max": float(timed.duration_s.max()),
                "fps_median": float(timed.fps.median()),
                "timing_conflicts": int((g.timing_conflict == True).sum()),
            }
        )
    overall = pd.DataFrame(rows)
    overall.to_csv(out / "tables/dataset_summary.csv", index=False)
    resolution = (
        df.groupby(["dataset", "format", "image_mode", "width", "height"], dropna=False)
        .agg(
            files=("case_id", "size"),
            frames_min=("n_frames", "min"),
            frames_median=("n_frames", "median"),
            frames_max=("n_frames", "max"),
            fps_median=("fps", "median"),
            duration_median=("duration_s", "median"),
        )
        .reset_index()
    )
    resolution["within_dataset_percent"] = (
        resolution.files / resolution.dataset.map(df.dataset.value_counts()) * 100
    )
    resolution.to_csv(out / "tables/format_resolution_distribution.csv", index=False)
    missing = []
    for src in SOURCES:
        g = df[df.dataset == src]
        checks = {
            "Original view label": g.original_view.notna(),
            "Verified patient ID": g.patient_id_known == True,
            "Frame count": g.n_frames.notna(),
            "Pixel dimensions": g.width.notna() & g.height.notna(),
            "Timing for dynamic objects": (
                g.loc[g.dynamic == True, "fps"].notna() & (g.loc[g.dynamic == True, "fps"] > 0)
            ),
            "Physical calibration": g.spatial_calibration_cm == True,
            "Expert transition labels": pd.Series(False, index=g.index),
            "Acquired ECG waveform": g.waveform_present == True,
        }
        for key, present in checks.items():
            missing.append(
                {
                    "dataset": src,
                    "field": key,
                    "unavailable": int((~present).sum()),
                    "denominator": len(present),
                    "unavailable_percent": float((~present).mean() * 100),
                }
            )
    missing = pd.DataFrame(missing)
    missing.to_csv(out / "tables/information_availability.csv", index=False)
    misstable = df.groupby("dataset").agg(
        {
            k: lambda s: int(s.isna().sum())
            for k in [
                "fps",
                "duration_s",
                "width",
                "height",
                "n_frames",
                "heart_rate",
                "ef",
                "edv",
                "esv",
                "original_view",
                "family_view",
            ]
        }
    )
    misstable.to_csv(out / "tables/field_null_counts.csv")
    splits = df.groupby(["dataset", "split"], dropna=False).size().reset_index(name="files")
    splits.to_csv(out / "tables/split_counts.csv", index=False)
    lengths = []
    for src in SOURCES:
        g = df[(df.dataset == src) & (df.dynamic == True)]
        if src == "MIMIC-IV-Echo":
            g = g[g.bmode_cine == True]
        x = g.duration_s.dropna()
        lengths.append(
            {
                "dataset": src,
                "n": len(x),
                "under_2s": int((x < 2).sum()),
                "from_2_to_5s": int(((x >= 2) & (x < 5)).sum()),
                "from_5_to_10s": int(((x >= 5) & (x < 10)).sum()),
                "at_least_10s": int((x >= 10).sum()),
                "median": float(x.median()),
                "p25": float(x.quantile(0.25)),
                "p75": float(x.quantile(0.75)),
                "p05": float(x.quantile(0.05)),
                "p95": float(x.quantile(0.95)),
                "maximum": float(x.max()),
            }
        )
    length_table = pd.DataFrame(lengths)
    length_table.to_csv(out / "tables/bmode_duration_summary.csv", index=False)
    return overall, resolution, missing, length_table


def distribution_figures(df, pred, qc, out, registry):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    ev, mi, en = [df[df.dataset == src] for src in SOURCES]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    view_names = {
        "A2C": "Apical two-chamber (A2C)",
        "A3C": "Apical three-chamber (A3C)",
        "A4C": "Apical four-chamber (A4C)",
        "A5C": "Apical five-chamber (A5C)",
        "PLAX": "Parasternal long-axis (PLAX)",
        "PSAX": "Parasternal short-axis (PSAX)",
        "SC4C": "Subcostal four-chamber (SC4C)",
        "SSN": "Suprasternal notch (SSN)",
        "Doppler PLAX": "Doppler: parasternal long-axis",
        "Doppler PSAX": "Doppler: parasternal short-axis",
        "Apical Doppler": "Apical Doppler",
        "Subcostal": "Subcostal",
        "Unmapped": "Unmapped (PMPALA)",
    }
    grouped_views = ev.family_view.fillna("Unmapped").value_counts().sort_values()
    axs[0].barh([view_names[k] for k in grouped_views.index], grouped_views.values, color=COLORS[0])
    axs[0].set_title("Cardiac views (EV9V)")
    pv = pred.predicted_view.value_counts().sort_values()
    axs[1].barh([view_names[k] for k in pv.index], pv.values, color=COLORS[1])
    axs[1].set_title("Predicted cardiac views (MIMIC-IV-Echo)")
    for ax in axs:
        ax.set_xlabel("Video")
        ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    savefig(
        fig,
        out,
        "01_views",
        "EV9V (n=5,138): released labels grouped into cardiac-view families. PASA, PMVLSA, PPMLSA and PMASA form PSAX (n=1,332); PMPALA remains unmapped (n=453). MIMIC (n=5,625): predicted classes from mean raw EchoPrime probabilities, not expert labels. EchoNet is omitted from this view-distribution figure.",
        "full EV9V inventory; existing MIMIC prediction caches",
        registry,
    )

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.5))
    ev2 = ev.assign(view=ev.family_view.fillna("Unmapped"))
    grouped = (
        ev2.groupby("view")
        .agg(videos=("case_id", "size"), seconds=("duration_s", "sum"))
        .sort_index()
    )
    grouped.to_csv(out / "tables/ev9v_view_counts_and_duration.csv")
    x = np.arange(len(grouped))
    axs[0].bar(
        x - 0.18,
        100 * grouped.videos / grouped.videos.sum(),
        0.36,
        label="Video share",
        color=COLORS[0],
    )
    axs[0].bar(
        x + 0.18,
        100 * grouped.seconds / grouped.seconds.sum(),
        0.36,
        label="Time share",
        color="#84babb",
    )
    axs[0].set_xticks(x, grouped.index)
    axs[0].set_ylabel("Within EV9V (%)")
    axs[0].legend()
    axs[0].set_title("Class imbalance changes with the denominator")
    # Dataset-by-view matrix keeps absent ontology support distinct from observed counts.
    labels = ["A4C", "A5C", "PLAX", "PSAX", "SC4C", "A2C", "A3C", "Other / unmapped"]

    def fam(name):
        return (
            {"Subcostal": "SC4C"}.get(name, name)
            if name in labels + ["Subcostal"]
            else "Other / unmapped"
        )

    ev_counts = ev2.view.map(fam).value_counts()
    mi_counts = pred.predicted_view.map(fam).value_counts()
    matrix = np.array(
        [
            [ev_counts.get(k, 0) / len(ev) * 100 for k in labels],
            [mi_counts.get(k, 0) / len(pred) * 100 for k in labels],
            [100 if k == "A4C" else 0 for k in labels],
        ]
    )
    axs[1].imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    axs[1].set_yticks(range(3), ["EV9V labels", "MIMIC predictions", "EchoNet scope"])
    axs[1].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    for i in range(3):
        for j in range(len(labels)):
            label = f"{matrix[i, j]:.1f}"
            if (i == 0 and labels[j] in ["A2C", "A3C"]) or (i == 2 and j != 0):
                label = "N/A"
            axs[1].text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 50 else "#17354c",
                fontsize=8,
            )
    axs[1].set_title("Source and view are strongly entangled (%)")
    fig.tight_layout()
    savefig(
        fig,
        out,
        "02_source_view",
        "N/A means outside the released label ontology or dataset scope, not proof that no such anatomical appearance occurs. PMPALA remains unmapped. The MIMIC denominator is cached predictions only.",
        "EV9V full manifest; EchoNet scope; MIMIC prediction subset",
        registry,
    )

    cohorts = [ev, mi[mi.bmode_cine == True], en]
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.5), sharex=True)
    bins = np.geomspace(0.1, 150, 38)
    for ax, src, g, color in zip(axs, SOURCES, cohorts, COLORS):
        vals = np.sort(g.duration_s.dropna().to_numpy())
        ax.hist(
            vals,
            bins=bins,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(f"{src} (n={len(vals):,})")
        ax.set_xscale("log")
        ax.set_xlim(0.1, 150)
        ax.set_xticks([0.5, 1, 2, 5, 10, 30, 100], ["0.5", "1", "2", "5", "10", "30", "100"])
        ax.set_xlabel("Video length")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    savefig(
        fig,
        out,
        "03_duration",
        "Video length is measured in seconds on a logarithmic axis; bars show video counts in identical bins, with a separate count scale for each dataset. EV9V and EchoNet include all videos; MIMIC includes dynamic 2D tissue loops. Playback duration is N/FPS. Stills and unknown-time objects are not assigned zero seconds.",
        "full inventories; MIMIC B-mode header selection",
        registry,
    )

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.6))
    for src, g, color in zip(SOURCES, [ev, mi, en], COLORS):
        sizes = g.groupby(["width", "height"]).size()
        axs[0].scatter(
            [a[0] for a in sizes.index],
            [a[1] for a in sizes.index],
            s=25 + np.sqrt(sizes.to_numpy()) * 6,
            alpha=0.65,
            label=src,
            color=color,
            edgecolor="white",
        )
        rates = np.sort(g.loc[g.dynamic == True, "fps"].dropna())
        axs[1].plot(rates, np.arange(1, len(rates) + 1) / len(rates), label=src, color=color)
    axs[0].set(
        xlabel="Original width (pixels)",
        ylabel="Original height (pixels)",
        title="Native canvas; marker area increases with count",
    )
    axs[0].legend(fontsize=8)
    axs[1].set(
        xlabel="Timing rate (frames / second)",
        ylabel="Cumulative fraction",
        title="Timing sources differ by dataset",
    )
    axs[1].legend(fontsize=8)
    axs[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout()
    savefig(
        fig,
        out,
        "04_size_fps",
        "Native pixel dimensions are not physical resolution. EV9V uses release playback FPS; EchoNet uses FileList FPS checked against AVI; MIMIC uses frame intervals where available. Marker sizes are descriptive, not proportional areas.",
        "all readable headers",
        registry,
    )

    missing = pd.read_csv(out / "tables/information_availability.csv")
    matrix = missing.pivot(index="field", columns="dataset", values="unavailable_percent")[SOURCES]
    fig, ax = plt.subplots(figsize=(10, 4.1))
    ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="Oranges")
    ax.set_xticks(range(3), SOURCES)
    ax.set_yticks(range(len(matrix)), matrix.index)
    for i in range(len(matrix)):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{matrix.iloc[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if matrix.iloc[i, j] > 65 else "#17354c",
            )
    ax.set_title("Information unavailable in the local files / tables")
    fig.tight_layout()
    savefig(
        fig,
        out,
        "05_missingness",
        "These are availability gaps, not all random missing values. Absent expert labels, patient linkage and physical calibration are structural. ECG waveforms exclude a trace drawn on the image. A missing view tag is not repaired by a model prediction.",
        "all input records; existing local annotation sources",
        registry,
    )

    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.4))
    for ax, key, title in zip(
        axs,
        ["brightness", "nonblack_fraction", "laplacian_variance"],
        ["Mean brightness", "Nonblack pixel fraction", "Laplacian variance"],
    ):
        vals = [qc.loc[(qc.dataset == s) & qc.decode_error.isna(), key].dropna() for s in SOURCES]
        b = ax.boxplot(
            vals, tick_labels=["EV9V", "MIMIC", "EchoNet"], showfliers=False, patch_artist=True
        )
        for patch, color in zip(b["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        if key == "laplacian_variance":
            ax.set_yscale("log")
    fig.tight_layout()
    savefig(
        fig,
        out,
        "06_quality",
        "Exploratory proxies from up to 16 ordered frames per sampled video, resized to 128x128 only for measurements. Equal allocation across strata is not population weighting. Background, acquisition and prior cropping confound cross-source comparisons; these are not clinical quality scores.",
        "fixed-seed stratified / patient-sampled frame QC",
        registry,
    )

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.5))
    axs[0].hist(en.ef, bins=np.arange(0, 105, 5), color=COLORS[2], edgecolor="white")
    axs[0].set(
        xlabel="Released ejection fraction (%)",
        ylabel="Videos",
        title="EchoNet measurements: descriptive context only",
    )
    hb = axs[1].hexbin(en.edv, en.esv, gridsize=45, mincnt=1, bins="log", cmap="Blues")
    fig.colorbar(hb, ax=axs[1], label="Videos (log color scale)")
    axs[1].set(
        xlabel="Released EDV (mL)",
        ylabel="Released ESV (mL)",
        title="Ventricular volume distribution",
    )
    fig.tight_layout()
    savefig(
        fig,
        out,
        "07_echonet_measurements",
        "EF/EDV/ESV are released measurements, not view or transition labels. These measurements do not establish view or transition ground truth. No disease classification or downstream model was run.",
        "EchoNet FileList, n=10,030",
        registry,
    )

    studies = pd.read_csv(out / "tables/mimic_study_downloads.csv")
    complete = studies[studies.complete == True]
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.5))
    axs[0].hist(complete.expected, bins=25, color=COLORS[1], edgecolor="white")
    axs[0].set(
        xlabel="DICOM objects per complete study",
        ylabel="Studies",
        title="Within-study correlation must be respected",
    )
    modes = mi.groupby(["image_mode", "dynamic"]).size()
    labels = [f"{a}\n" + ("dynamic" if b else "static") for a, b in modes.index]
    axs[1].barh(labels, modes.values, color=COLORS[1])
    axs[1].set(xlabel="DICOM objects", title="Modes from ultrasound-region metadata")
    fig.tight_layout()
    savefig(
        fig,
        out,
        "08_mimic_structure",
        "All locally downloaded DICOM objects remain in the inventory. The main motion-analysis cohort is dynamic tissue-only imaging. Other/mixed region codes are retained as a separate group; complete-study selection changes the denominator.",
        "frozen official record-list match and DICOM headers",
        registry,
    )


def mask_examples(df, out, registry):
    chosen = [("ev9v_d47d8e7414433eb2", "Normal fan"), ("ev9v_a9b7d8c2e96c23f1", "Dark sector")]
    # Resolve the original source from the frozen 42-case manifest, not this script's ID scheme.
    legacy = pd.read_csv("configs/audit_manifest.csv")
    records = []
    for old_id, title in chosen:
        old = legacy[legacy.id == old_id].iloc[0]
        name = Path(old.path).stem
        candidates = df[(df.dataset == "EV9V") & (df.original_id == name)]
        if candidates.empty:
            raise ValueError(f"Cannot resolve approved example {old_id}")
        r = candidates.iloc[0].to_dict()
        n = int(r["n_frames"])
        # Main examples must use the inventoried MP4, not its JPEG derivative.
        video_row = {**r, "dataset": "EV9V-native-MP4"}
        frames = frames_for(video_row, range(n))
        sector = detect_sector(frames, exclusion_mask=ev9v_hint_mask(frames.shape[1:3]))
        cine = Cine(frames, float(r["fps"]), r["path"], dataset="EV9V")
        clean, record = standardize_cine(cine, sector)
        rows = np.linspace(0, n - 1, 3).round().astype(int)
        fig, axs = plt.subplots(3, 4, figsize=(12.4, 6.8))
        motion = frames.astype(float).std(axis=0).mean(axis=-1)
        for j, k in enumerate(rows):
            axs[j, 0].imshow(frames[k])
            axs[j, 1].imshow(motion, cmap="magma")
            axs[j, 2].imshow(frames[k])
            axs[j, 2].contour(sector.mask, levels=[0.5], colors=["#00e27c"], linewidths=0.8)
            axs[j, 3].imshow(clean.frames[k])
            for ax in axs[j]:
                ax.set_xticks([])
                ax.set_yticks([])
            axs[j, 0].set_ylabel(f"Frame {k}\n{k / r['fps']:.2f} s")
        for ax, label in zip(
            axs[0],
            ["Original", "Dynamic pixel method", "Fixed proposed mask", "Preprocessed output"],
        ):
            ax.set_title(label)
        fig.suptitle(f"{title}: {sector.status} | {record['action']}", fontsize=14)
        fig.tight_layout()
        name = "09_mask_normal" if title == "Normal fan" else "10_mask_dark"
        savefig(
            fig,
            out,
            name,
            "One mask is inferred from temporal evidence and held fixed through the cine. The dark case retains original frames: the displayed proposal is not an accepted crop. EV9V: Bo Gou et al., CC BY 4.0; project-generated visualization.",
            "existing EV9V examples; unchanged current detector, full native frame order",
            registry,
        )
        if title == "Normal fan":
            jpeg_frames = frames_for(r, range(n))
            jpeg_sector = detect_sector(
                jpeg_frames, exclusion_mask=ev9v_hint_mask(jpeg_frames.shape[1:3])
            )
            jpeg_status = jpeg_sector.status
        else:
            jpeg_status = None
        records.append(
            {
                "case_id": r["case_id"],
                "example": title,
                "status": sector.status,
                "action": record["action"],
                "input_width": cine.width,
                "input_height": cine.height,
                "output_width": clean.width,
                "output_height": clean.height,
                "n_frames": n,
                "mask_fraction": float(sector.mask.mean()),
                "geometry_family": sector.stats.get("stage2_family"),
                "fixed_per_cine": True,
                "pixel_source": "native MP4",
                "matched_jpeg_status": jpeg_status,
            }
        )
    pd.DataFrame(records).to_csv(out / "tables/masking_examples.csv", index=False)


def temporal_example(df, routing, out, registry, vid, name, title, uncertain=False):
    r = df[(df.dataset == "EV9V") & (df.original_id == vid)].iloc[0].to_dict()
    p = routing / "runs/_demo/realvideo/probs_sorted" / f"{vid}.npz"
    with np.load(p) as z:
        t = z["t_sec"]
        indices = z["frame_idx"]
        pb = z["prob_b0"]
        pe = z["prob_ep"]
    if not np.all(np.diff(indices) > 0) or not np.allclose(t, indices / float(r["fps"]), atol=1e-4):
        raise ValueError("Temporal cache fails frame-index / time validation")
    n = int(r["n_frames"])
    fps = float(r["fps"])
    if uncertain:
        entropy = -(pe * np.log(np.maximum(pe, 1e-8))).sum(axis=1)
        smooth = np.convolve(entropy, np.ones(5) / 5, mode="same")
        center = float(t[int(smooth.argmax())])
        lo = max(0, center - 0.6)
        hi = min(float(t[-1]), center + 0.6)
        picks = np.linspace(lo * fps, hi * fps, 6).round().astype(int)
    else:
        screen = pd.read_csv(routing / "runs/_demo/realvideo/screen_ev9v.csv")
        center = float(
            screen[(screen.video_id == vid) & (screen.recognizer == "EchoPrime")].split_s.iloc[0]
        )
        lo = max(0, center - 0.5)
        hi = min(float(t[-1]), center + 0.5)
        picks = np.linspace(0, n - 1, 6).round().astype(int)
    imgs = frames_for(r, picks)
    fig = plt.figure(figsize=(12.4, 6.3))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1])
    for j, (k, img) in enumerate(zip(picks, imgs)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{k / fps:.2f} s | f{k}", fontsize=9)
    for row, (probs, model, names) in enumerate(
        [(pb, "B0 (EV9V-trained)", FAMILY5), (pe, "EchoPrime", FAMILY5 + ["Other"])], 1
    ):
        ax = fig.add_subplot(gs[row, :])
        for j, label in enumerate(names):
            if probs[:, j].max() >= 0.15:
                ax.plot(
                    t,
                    probs[:, j],
                    label=label,
                    linewidth=1.4,
                    color={
                        "PLAX": "#4769a5",
                        "PSAX": "#9b5db5",
                        "A4C": "#147d80",
                        "A5C": "#c88722",
                        "SC4C": "#ba5654",
                        "Other": "#697680",
                    }[label],
                )
        ax.axvspan(lo, hi, color="#697680", alpha=0.15, label="Review interval")
        ax.set(
            xlim=(0, (n - 1) / fps),
            ylim=(0, 1),
            ylabel="Baseline probability" if row == 1 else "EchoPrime probability",
            xlabel="Video time (s)",
        )
        ax.set_title(model, loc="left", fontsize=10)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    fig.suptitle(title + "\nCandidate only; no independent expert segment labels", fontsize=13)
    fig.tight_layout()
    savefig(
        fig,
        out,
        name,
        "Chronological frames from one native EV9V video. The upper plot shows baseline (B0) probabilities and the lower plot shows EchoPrime probabilities; video time is in seconds. Shading marks a review interval, not a verified transition boundary. Prediction changes do not establish probe movement. EV9V: Bo Gou et al., CC BY 4.0.",
        "purposefully selected discovery example; not a prevalence sample",
        registry,
    )
    return {
        "case_id": r["case_id"],
        "video_id": vid,
        "start_s": lo,
        "end_s": hi,
        "candidate_kind": "uncertain appearance / possible probe motion"
        if uncertain
        else "possible A4C-to-A5C change",
        "selection": "highest smoothed EchoPrime entropy"
        if uncertain
        else "previous EchoPrime split candidate",
        "annotation_source": "model-screened; assistant visual review only",
        "expert_verified": False,
        "review_status": "pending independent human review",
        "native_single_file": True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--routing-root", type=Path, default=Path("/home/william/echo-view-routing"))
    args = ap.parse_args()
    out = args.out
    routing = args.routing_root
    df = pd.read_csv(
        out / "tables/video_inventory.csv",
        low_memory=False,
        dtype={"subject_id": str, "study_id": str},
    )
    assert df.case_id.is_unique
    pred = predictions(df, routing, out)
    print("MIMIC prediction matches", len(pred), flush=True)
    qc = qc_sample(df, out)
    # Match the on-disk representation, where empty errors are NaN.
    qc = pd.read_csv(out / "tables/qc_sample_results.csv")
    summaries(df, pred, out)
    registry = []
    distribution_figures(df, pred, qc, out, registry)
    mask_examples(df, out, registry)
    segments = []
    segments.append(
        temporal_example(
            df,
            routing,
            out,
            registry,
            "2022-07-10_10-58-17_320x240_7",
            "11_transition",
            "Within-video change candidate: A4C / A5C",
        )
    )
    segments.append(
        temporal_example(
            df,
            routing,
            out,
            registry,
            "2022-05-24_14-30-39_320x240_1",
            "12_uncertain",
            "Uncertain interval: possible view change or limited evidence",
            True,
        )
    )
    pd.DataFrame(segments).to_csv(out / "tables/segment_inventory.csv", index=False)
    (out / "figure_manifest.json").write_text(json.dumps(registry, indent=2))
    (out / "analysis_provenance.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "models_run": False,
                "training_run": False,
                "prediction_cache": "existing corrected numeric-order EV9V cache; existing MIMIC EchoPrime cache",
                "input_inventory_sha256": hashlib.sha256(
                    (out / "tables/video_inventory.csv").read_bytes()
                ).hexdigest(),
                "sampling": "EV9V label x duration, EchoNet split x duration, MIMIC one B-mode cine per sampled patient",
                "image_measurement_size": [128, 128],
                "max_frames_per_qc_cine": 16,
                "qc_sample_n": len(qc),
                "qc_sample_by_source": qc.dataset.value_counts().to_dict(),
                "qc_decode_errors": int(qc.decode_error.notna().sum()),
                "mimic_predictions_n": len(pred),
                "figures": len(registry),
                "independent_expert_labels": False,
            },
            indent=2,
        )
    )
    print("ANALYSIS COMPLETE", len(registry), "figures", flush=True)


if __name__ == "__main__":
    main()
