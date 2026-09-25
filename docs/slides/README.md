# LaTeX 阶段汇报幻灯片

英文方法与结果汇报见 [README_EN.md](README_EN.md)：按数据、方法、实验和结果组织，不使用 Stage 编号。

`progress_2026-09-25.tex`：11 页、16:9、中文 Beamer 幻灯片。内容按数据、两阶段方法、具体实验、定量结果、实际案例、视觉质检和下一步组织。

汇报日期为 2026-09-25；实验结果截至 2026-09-18，实验代码版本为 `c2289a9`。内容依据 [`PROGRESS_REPORT_2026-09-25_ZH.md`](../PROGRESS_REPORT_2026-09-25_ZH.md) 及其列出的冻结审计表，本次未重跑预处理实验。5 个条目接受几何拟合，整体视觉评级尚未提升；稳定性 IoU 不是人工真值分割精度。

## 编译

在 `.tex` 所在目录执行以下任一方式：

```bash
tectonic --keep-logs progress_2026-09-25.tex
```

或使用带 Beamer、fontspec、xeCJK、TikZ 的 TeX Live，运行 XeLaTeX 两次：

```bash
xelatex progress_2026-09-25.tex
xelatex progress_2026-09-25.tex
```

本次使用 Tectonic 0.17.0 编译。Overleaf 可上传本地源码 ZIP，并选择 XeLaTeX 和主文件 `progress_2026-09-25.tex`。

- 本地演示包含 `fonts/NotoSansSC-Regular.ttf`、`NotoSansSC-Bold.ttf` 和 OFL 字体许可。它们由 Google Fonts Noto Sans SC 可变字体分别以 400、650 字重生成。
- 缺少本地字体时，源码使用 TeX 自带的 TeX Gyre Heros 和 Fandol Hei。
- 缺少本地案例图时，两张案例页显示占位框。无本地字体和案例图的公开源码也已完成 11 页编译检查。

## 案例图片

本地演示使用以下既有 Stage 2 四联图，无新增数据实验：

| 演示包文件 | 本地审计输出 | 内容 |
|---|---|---|
| `assets/normal_fan.png` | `outputs/stage2_subset_release/cases/ev9v_d47d8e7414433eb2.png` | A4C，接受标准扇形拟合 |
| `assets/dark_sector.png` | `outputs/stage2_subset_release/cases/ev9v_a9b7d8c2e96c23f1.png` | PASA 暗区，低置信度保留 |

图片来源：[EV9V，Bo Gou 等](https://huggingface.co/datasets/bgx666/EV9V)，数据集许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。修改包括候选运动证据、界面排除区域、几何轮廓、处理后图像和四联图标注；不构成临床验证。图片许可独立于仓库代码许可。

按仓库发布边界，以上案例图、含图 PDF/PPTX 及 ZIP 保留在忽略的 `outputs/slides_2026-09-25/`；公开仓库仅提交文本源码、说明和导出脚本。

## PowerPoint 与交付文件

LaTeX PDF 保留文字及原生矢量图。为方便 PowerPoint 放映，另提供 1920 像素宽的整页图像版 PPTX；**PPTX 内的文字和图表不是独立可编辑对象，修改内容请编辑 `.tex` 后重新编译。** 每页备注保留 PDF 提取文字。

在安装 `pymupdf`、`python-pptx` 后，可用本目录脚本从最终 PDF 生成 PPTX：

```bash
python export_deck.py progress_2026-09-25.pdf --outdir ./delivery
```

本地交付文件：

- `Echo_Preprocessing_2026-09-25.pdf`：完整 11 页 PDF。
- `Echo_Preprocessing_2026-09-25.pptx`：11 页 PowerPoint 放映版。
- `Echo_Preprocessing_LaTeX_Package_2026-09-25.zip`：源码、字体及许可、两张案例图、说明、导出脚本和 PDF/PPTX。

已核对数据计数和关键结果，检查 11 页渲染、16:9 尺寸、TeX 溢出日志、PPTX 页数和 ZIP 完整性。文档修改未改变预处理代码。
