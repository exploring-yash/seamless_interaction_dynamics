# Seamless Interaction Dynamics

Analysis of interpersonal synchrony and coupling dynamics using [Meta's Seamless Interaction dataset](https://huggingface.co/datasets/facebook/seamless-interaction) (16,267 conversations, 4,285 participants, 30 Hz behavioral signals).

## Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| [seamless_data_pipeline.ipynb](Notebooks/seamless_data_pipeline.ipynb) | Download and organize balanced stranger/familiar dyads from HuggingFace | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exploring-yash/seamless_interaction_dynamics/blob/main/Notebooks/seamless_data_pipeline.ipynb) |
| [measuring_synchrony.ipynb](Notebooks/measuring_synchrony.ipynb) | Multi-modal coupling measurement: PSD diagnostic → CCA on body + FAU + head → partner-shuffled null → four-model classification ablation. Establishes the measurement layer; reports a labelled `[NEGATIVE RESULT]` for the familiar-vs-stranger H1 with full transparency on the FAU-dominance and T-confound caveats. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exploring-yash/seamless_interaction_dynamics/blob/main/Notebooks/measuring_synchrony.ipynb) |
| [early_warning_signals.ipynb](Notebooks/early_warning_signals.ipynb) | Critical Slowing Down (CSD) on the multimodal coupling signal `CCA_1(t)` from the previous notebook. Tests whether variance / AR(1) / skewness rise before independently-defined head + gaze rupture events using a circular block-bootstrap null. Adds four diagnostic probes: KSG mutual-information vs CCA linearity test, calibrated-FPR rupture detector (Boettiger-Hastings 2012), per-modality CCA decomposition with duration-confound audit, and nested-CV alternative classifiers. Reports a labelled H1 verdict (`[NO EFFECT vs NULL]` at the sample-run cohort) **alongside** a per-modality classification result that survives the duration audit at AUC = 0.778 (`[EMPIRICAL]`). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exploring-yash/seamless_interaction_dynamics/blob/main/Notebooks/early_warning_signals.ipynb) |

## Dataset

- **Source:** [facebook/seamless-interaction](https://huggingface.co/datasets/facebook/seamless-interaction) (HuggingFace)
- **GitHub:** [facebookresearch/seamless_interaction](https://github.com/facebookresearch/seamless_interaction)
- **License:** BSD-style
- **Features:** Emotion valence/arousal, 24-dim Facial Action Units, gaze, head pose, body translation (30 Hz)
- **Relationship labels:** Stranger vs. familiar (friends, coworkers, family, etc.)

## Repository layout

```
.
├── Notebooks/
│   ├── seamless_data_pipeline.ipynb     # Download + organize NPZ feature files
│   ├── measuring_synchrony.ipynb        # CCA-based multimodal coupling pipeline
│   └── early_warning_signals.ipynb      # CSD on CCA_1(t) + 4 diagnostic probes (MI, calibrated-FPR, per-modality CCA, alt classifiers)
├── experiments/
│   └── signal_utils.py                  # Shared signal-processing backbone (~2,200 LOC)
├── results/                             # Deliverable artifacts from a sample run
│   ├── cca_features.csv                 # 186 dyad × 14 feature row-table (NB measuring_synchrony)
│   ├── psd_diagnostic.json              # Per-modality bandpass + SNR (NB measuring_synchrony)
│   ├── qc_pass_core.json                # Dyads passing the quality-gate (NB measuring_synchrony)
│   ├── section5_1_summary.json          # Per-channel R(t) AUCs (NB measuring_synchrony)
│   ├── step4_summary.json               # Primary-model AUC + label (NB measuring_synchrony output → consumed by NB early_warning_signals)
│   ├── h1_verdict_summary.json          # H1 verdict + merged §7/§8/§9/§10 sidecars (NB early_warning_signals primary output)
│   ├── mi_diagnostic_summary.json       # KSG MI vs CCA linearity gap (NB early_warning_signals §7)
│   ├── per_modality_cca_summary.json    # Per-modality CCA AUC + duration-residualized AUC + Hanley-McNeil 95% CI (NB early_warning_signals §9)
│   ├── alt_classifiers_summary.json     # Nested-CV across LR/HGB/SVM with bootstrap CIs (NB early_warning_signals §10)
│   ├── cca_diagnostic.png               # 4-panel CCA diagnostic figure (NB measuring_synchrony)
│   └── section5_2_garijo_kc_hist.png    # Critical-coupling K_c histogram (NB measuring_synchrony)
├── requirements.txt
└── README.md
```

## Quick Start

1. Open `seamless_data_pipeline.ipynb` in Colab (badge above) — downloads ~200 balanced dyads (~12 GB) to your Google Drive.
2. Open `measuring_synchrony.ipynb` in Colab (badge above) — runs the full PSD → CCA → classification pipeline against the dataset you downloaded in step 1; writes `step4_summary.json` and `features_df.pkl` to `results/`.
3. Open `early_warning_signals.ipynb` in Colab (badge above) — reads the artifacts from step 2 and runs CSD on the multimodal coupling signal `CCA_1(t)` plus four diagnostic probes (MI vs CCA, calibrated-FPR rupture detector, per-modality CCA with duration audit, alternative classifiers). Writes the H1 verdict and §7–§10 sidecars to `results/`.
4. The `results/` folder contains a sample run's output artifacts (~620 KB) so you can preview the deliverable shape before running anything.

For local execution (outside Colab):

```bash
pip install -r requirements.txt
jupyter lab Notebooks/measuring_synchrony.ipynb
# or
jupyter lab Notebooks/early_warning_signals.ipynb
```

## Requirements

- Google Colab (recommended) or local Python 3.10+
- [HuggingFace account](https://huggingface.co/join) with [dataset access](https://huggingface.co/datasets/facebook/seamless-interaction) approved
- ~12 GB Google Drive space (for Full feature tier)
- Pinned Python deps: see `requirements.txt`
