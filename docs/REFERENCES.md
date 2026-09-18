# References and reusable code

## Papers

- **Naser JA et al. (2024). Artificial intelligence-based classification of echocardiographic views.**
  Eur Heart J Digit Health 5(3):260-269. doi:10.1093/ehjdh/ztae015.
  https://academic.oup.com/ehjdh/article/5/3/260/7614559 (open access at PMC11104471).
  Preprocessing, quoted: DICOMs were "cropped in an automated way to show the echocardiographic window";
  "the changing portion of the video across the temporal dimension ... was identified, and morphological
  transformations were applied. Finally, a bounding box was drawn around the largest moving portion of
  the video in order to define the imaging sector." ECG "was removed if it was outside the sector but
  was retained if it overlapped the imaging sector". First second discarded, then 10 consecutive frames
  from the middle of each cine; resized to 256x256 with padding; pydicom 2.3 + opencv 4.5.
  Exclusions: < 10 frames (incl. Doppler/M-mode stills), congenital protocols, cines whose view changes.
- **Ouyang D et al. (2020). Video-based AI for beat-to-beat assessment of cardiac function (EchoNet-Dynamic).** Nature 580:252-256.
- **Gou B et al. Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos (STFM / EV9V).** arXiv:2606.17437. Dataset: https://huggingface.co/datasets/bgx666/EV9V (CC-BY-4.0).
- **Vukadinovic M et al. (2024). EchoPrime.** arXiv:2410.09704. Code: https://github.com/echonet/EchoPrime.
- **EchoViewCLIP.** Code: https://github.com/xmed-lab/EchoViewCLIP.
- **Leclerc S et al. (2019). CAMUS.** IEEE TMI 38(9). https://www.creatis.insa-lyon.fr/Challenge/camus/
- **MIMIC-IV-Echo v1.0.1.** https://physionet.org/content/mimic-iv-echo/1.0.1/ (credentialed; 524,137 DICOMs, 7,228 studies, 4,572 patients; path layout `files/pNN/pXXXXXXXX/sZZZZZZZZ/ZZZZZZZZ_VVVV`).
- **HMC-QU** (A4C/A2C, Philips + GE, 25 fps, 422x636 to 768x1024, raw screens): https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset.
- **Echo-Toolkit** (neural-network sector prediction + RANSAC geometric fit): could not be located by
  name on GitHub, PyPI, arXiv or Semantic Scholar on 2026-09-17. Ask the mentor for the exact link.

## Code worth reusing or comparing against

| Where | What | Notes |
|---|---|---|
| `echonet/dynamic/scripts/ConvertDICOMToAVI.ipynb` | DICOM -> AVI, fixed *geometric* diagonal-band mask, crop dark rows, square, 10 % margin, 112x112 MJPEG, fps from (0018,0040) | Not adaptive: assumes the Stanford screen layout. Useful only as a format reference. |
| `EchoPrime/utils/utils.py::mask_outside_ultrasound` (vendored in `echo-view-routing/third_party/EchoPrime`) | "Ever non-zero" sum map eroded 10x (kills ECG/text) intersected with first-vs-last-frame difference, dilated 10x, flood fill, convex hull, mask applied to every frame | Closest public implementation of the Mayo idea. Weak points: uses only 2 frames for motion; fixed 3x3 kernel x 10 iterations regardless of resolution; erodes away thin sectors at 112 px. Our detector generalizes this (temporal std over up to 64 frames, resolution-scaled kernels, support-region union). |
| `EchoPrime/utils/utils.py::crop_and_scale` | aspect crop + 10 % zoom + resize to 224 | EchoPrime's model-side resize; belongs to the consumer, not the canonical store. |
| pydicom 3 `SequenceOfUltrasoundRegions` (0018,6011) | Region bounding boxes (`RegionLocationMinX0` ...), data type (tissue / colour flow / spectral / ECG wave) and physical pixel spacing | Free sector hint for DICOM sources when vendors fill it in; our DICOM reader already extracts it into `cine.meta["us_regions"]`. |
| EchoJEPA (arXiv:2602.02603) | Idealized sector mask from apex, half-angle, max radius | Geometric parametrization we can fit to our hull (RANSAC-style) as the next refinement. |
