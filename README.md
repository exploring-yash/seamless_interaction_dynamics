# Seamless Interaction Dynamics

Analysis of interpersonal synchrony and coupling dynamics using [Meta's Seamless Interaction dataset](https://huggingface.co/datasets/facebook/seamless-interaction) (16,267 conversations, 4,285 participants, 30 Hz behavioral signals).

## Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| [seamless_data_pipeline.ipynb](Notebooks/seamless_data_pipeline.ipynb) | Download and organize balanced stranger/familiar dyads from HuggingFace | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/exploring-yash/seamless_interaction_dynamics/blob/main/Notebooks/seamless_data_pipeline.ipynb) |

## Dataset

- **Source:** [facebook/seamless-interaction](https://huggingface.co/datasets/facebook/seamless-interaction) (HuggingFace)
- **GitHub:** [facebookresearch/seamless_interaction](https://github.com/facebookresearch/seamless_interaction)
- **License:** BSD-style
- **Features:** Emotion valence/arousal, 24-dim Facial Action Units, gaze, head pose, body translation (30 Hz)
- **Relationship labels:** Stranger vs. familiar (friends, coworkers, family, etc.)

## Quick Start

1. Open the data pipeline notebook in Colab (click the badge above)
2. Run all cells -- the pipeline downloads ~200 balanced dyads (~12 GB) to Google Drive
3. Use the output NPZ files for downstream analysis

## Requirements

- Google Colab (recommended) or local Python 3.10+
- [HuggingFace account](https://huggingface.co/join) with [dataset access](https://huggingface.co/datasets/facebook/seamless-interaction) approved
- ~12 GB Google Drive space (for Full feature tier)
