# Seamless Interaction Dynamics

Analysis of interpersonal synchrony and coupling dynamics using [Meta's Seamless Interaction dataset](https://huggingface.co/datasets/facebook/seamless-interaction) (16,267 conversations, 4,285 participants, 30 Hz behavioral signals).

## Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| [seamless_data_pipeline.ipynb](Notebooks/seamless_data_pipeline.ipynb) | Download and organize balanced stranger/familiar dyads from HuggingFace | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exploring-yash/seamless_interaction_dynamics/blob/main/Notebooks/seamless_data_pipeline.ipynb) |
| [measuring_synchrony.ipynb](Notebooks/measuring_synchrony.ipynb) | Multi-modal coupling measurement: PSD diagnostic → CCA on body + FAU + head → partner-shuffled null → four-model classification ablation. Establishes the measurement layer; reports a labelled `[NEGATIVE RESULT]` for the familiar-vs-stranger H1 with full transparency on the FAU-dominance and T-confound caveats. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exploring-yash/seamless_interaction_dynamics/blob/main/Notebooks/measuring_synchrony.ipynb) |

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
│   └── measuring_synchrony.ipynb        # CCA-based multimodal coupling pipeline
├── experiments/
│   └── signal_utils.py                  # Shared signal-processing backbone (~2,200 LOC)
├── results/                             # Deliverable artifacts from a sample run
│   ├── cca_features.csv                 # 186 dyad × 14 feature row-table
│   ├── psd_diagnostic.json              # Per-modality bandpass + SNR
│   ├── qc_pass_core.json                # Dyads passing the quality-gate
│   ├── section5_1_summary.json          # Per-channel R(t) AUCs
│   ├── cca_diagnostic.png               # 4-panel CCA diagnostic figure
│   └── section5_2_garijo_kc_hist.png    # Critical-coupling K_c histogram
├── requirements.txt
└── README.md
```

## Quick Start

1. Open `seamless_data_pipeline.ipynb` in Colab (badge above) — downloads ~200 balanced dyads (~12 GB) to your Google Drive.
2. Open `measuring_synchrony.ipynb` in Colab (badge above) — runs the full PSD → CCA → classification pipeline against the dataset you downloaded in step 1.
3. The `results/` folder contains a sample run's output artifacts (~308 KB) so you can preview the deliverable shape before running anything.

For local execution (outside Colab):

```bash
pip install -r requirements.txt
jupyter lab Notebooks/measuring_synchrony.ipynb
```

## Requirements

- Google Colab (recommended) or local Python 3.10+
- [HuggingFace account](https://huggingface.co/join) with [dataset access](https://huggingface.co/datasets/facebook/seamless-interaction) approved
- ~12 GB Google Drive space (for Full feature tier)
- Pinned Python deps: see `requirements.txt`
