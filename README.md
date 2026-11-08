# Continual-Medical-Image-Classification-on-BMAD


**Preliminary research code and technical report — November 2026**

**Researcher:** Peter Chika Ozo-ogueji (American University)  
**Faculty mentor:** Prof. Roberto Corizzo (American University)

This project investigates **supervised domain-incremental continual learning** for medical image classification, using [Mammoth](https://github.com/aimagelab/mammoth). It follows the six-domain sequence proposed for medical continual anomaly detection by [Barusco et al.](https://doi.org/10.1145/3746259.3760434), but **asks a different question**: can a classifier trained on labeled *normal and abnormal* images learn sequential medical domains without catastrophic forgetting?

The results are described in the [[preliminary technical report](https://github.com/PeterOzo/Continual-Medical-Image-Classification-on-BMAD/blob/main/BMAD_Continual_Learning_Report_Peter_Ozo-ogueji%20(4).pdf]. **This is not a peer-reviewed publication and is not the PyCLAD anomaly-detection benchmark.** An extension to continual visual anomaly detection is planned, not reported here as completed.

## Research question and task sequence

Each task uses the same binary output labels (normal versus abnormal); the imaging distribution changes by anatomy, modality, or source. The order is:

1. Brain MRI (BraTS 2021)
2. Liver CT (LiTS)
3. Retina OCT (RESC)
4. Chest X-ray (RSNA Pneumonia)
5. Histopathology (Camelyon16)
6. Retina OCT (OCT2017)

**Setting:** domain-incremental learning; a shared two-logit classifier is evaluated on all tasks seen so far after each task. This stream is simulated from distinct datasets; it does **not** demonstrate deployment across real hospitals.

## Preliminary results

The report compares fine-tuning, Experience Replay (ER), Dark Experience Replay++ (DER++), Averaged Gradient Episodic Memory (A-GEM), online Elastic Weight Consolidation (EWC), and Synaptic Intelligence (SI). Joint training and separate models per task are references, not deployable continual-learning baselines.

| Method | Final average image AUROC (3 seeds) | 95% t-interval half-width |
|---|---:|---:|
| Joint training (offline reference) | 0.966 | 0.006 |
| Per-task models (reference) | 0.929 | 0.074 |
| DER++ | 0.915 | 0.034 |
| ER | 0.914 | 0.036 |
| A-GEM | 0.722 | 0.133 |
| SI | 0.658 | 0.115 |
| Online EWC | 0.648 | 0.121 |
| Fine-tuning | 0.612 | 0.033 |

These numbers are **transcribed from Table 2 of the attached report**. They are not freshly recomputed here because the raw run logs and BMAD image files are not part of this repository. See [reports/table2_summary.csv](reports/table2_summary.csv) and the report for the full metrics, paired comparisons, per-task matrices, and limitations. AUROC is threshold-independent and does not measure clinical safety.

## Experimental protocol

- **Data:** BMAD images pooled and re-split into balanced normal/abnormal train, validation, and test sets; group-aware split and exact-duplicate audits. Details and important caveats: [docs/DATA.md](docs/DATA.md).
- **Architecture:** ImageNet-pretrained ResNet-18, 224 × 224 RGB images.
- **Training:** SGD, learning rate 0.01, momentum 0.9, batch size 32, five epochs per task; a 200-image reservoir memory for replay methods.
- **Tuning:** method-specific hyperparameters selected on validation data (seed 0); joint-training reference also tuned on validation.
- **Evaluation:** three training seeds (0, 1, 2) on a fixed data split; final average AUROC, stream average, backward transfer, forgetting, accuracy and F1; 95% t-intervals and group-level bootstrap analysis.
- **Additional analyses:** two alternative task orders (single seed), image-statistics shortcut audit, source-confound checks, and instability/failure reporting.

The report identifies important limitations: one split, nondeterministic GPU behavior, approximate Camelyon16 slide grouping, only ten Liver CT volumes, and small tuning budgets. **Do not interpret results as clinical validation or as directly comparable with normal-only anomaly-detection studies.**

## Repository structure

```text
src/bmad_cl/          Dataset indexing, split audits, experiment runner, statistics, reporting
mammoth_plugin/       Custom seq-bmad dataset plugin for Mammoth
scripts/              Command-line experiment/report drivers
tests/                89 synthetic-data unit tests
notebooks/            Output-cleared Colab experiment driver
docs/DATA.md          Dataset sources, grouping rules, licensing cautions
reports/              Preliminary technical report and transcribed summary metrics
.github/workflows/    Lint and unit-test CI
```

## How to run

**1. Obtain the data yourself.** Download BMAD from the [original BMAD repository](https://github.com/DorisBao/BMAD) and follow its source-dataset access conditions. This repository does not distribute medical images or raw patient data. The Colab notebook expects a resized data cache in your Drive; change `DRIVE_DIR` to your own folder.

**2. Colab:** open [notebooks/BMAD_CL_Colab_v4_1.ipynb](notebooks/BMAD_CL_Colab_v4_1.ipynb), choose a GPU runtime, configure your Drive path and run switches, then run the notebook sequentially. The notebook clones this repository (or extracts a ZIP at `REPO_ZIP`), installs Mammoth at pinned commit `e75a491c69fd729edeb01431afb753d9157d9a81`, and runs the tests before training. If the repository is private, supply a local ZIP in Drive rather than embedding GitHub credentials in the notebook.

**3. Command-line workflow:**

```bash
python -m pip install -e '.[report]'
python -m pip install -e '.[dev]'
MPLBACKEND=Agg python -m pytest -q
# Clone Mammoth separately and check out the pinned commit; install its dependencies.
cp mammoth_plugin/seq_bmad.py /path/to/mammoth/datasets/
python scripts/run_experiments.py --mammoth /path/to/mammoth --bmad-root /path/to/BMAD_224 --out /path/to/output --dry-run
```

**Important:** the notebook is the most complete driver for the report's protocol. The CLI is a starting point and does not automatically run every notebook-only stage (for example, joint-reference tuning and all sensitivity analyses). A GPU and the original datasets are required for full reproduction.

## Tests and provenance

The source extracted from the uploaded notebook passed **89 unit tests** in a local Python environment with `PYTHONPATH=src` and `pytest`. These tests cover logic on synthetic data; they **do not rerun the medical-image training or independently verify the report's experimental results**. CI is configured for Python 3.10 and 3.12. The notebook strips outputs before public sharing to avoid distributing sample images, Drive paths from outputs, or incidental metadata.

## Attribution, licensing and planned work

This is Peter Chika Ozo-ogueji's preliminary research under Prof. Roberto Corizzo's faculty mentorship; it is **not** presented as a joint publication or as code authored by the professor. As acknowledged in the report, an AI coding assistant (Anthropic Claude) was used in code development. Mammoth, BMAD and all source datasets remain credited to their original authors.

The next phase being discussed is **continual medical visual anomaly detection** (normal-only training and normal/abnormal testing), using the development GitHub version of [PyCLAD](https://github.com/lifelonglab/pyCLAD), with possible cross-dataset/domain-transfer studies. These are proposals, not completed experiments.

**No license is granted for this repository at present.** Confirm ownership, institutional requirements, and source-data licensing with Prof. Corizzo/American University before publishing or assigning an open-source license. The report contains example images from external datasets; confirm whether the report may be shared publicly under their terms.

### References

- [BMAD: Benchmarks for Medical Anomaly Detection](https://arxiv.org/abs/2306.11876)
- [Barusco et al., Towards Continual Visual Anomaly Detection in the Medical Domain](https://doi.org/10.1145/3746259.3760434)
- [Mammoth continual-learning framework](https://github.com/aimagelab/mammoth)
- [PyCLAD continual anomaly-detection framework](https://github.com/lifelonglab/pyCLAD)
