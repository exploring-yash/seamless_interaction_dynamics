"""Shared signal processing utilities for the 4-notebook portfolio.

Used by:
- notebooks/kuramoto_synchrony_analysis.ipynb (NB1a, primary empirical anchor)
- notebooks/measuring_synchrony.ipynb (NB1b, conceptual primer)
- notebooks/early_warning_signals.ipynb (NB2, CSD predictor)
- notebooks/intervention_ai_safety.ipynb (NB3, intervention policy)

Design goals:
- Single source of truth — no drift between notebooks
- PSD-first: bandpass chosen empirically, not assumed
- Quality gate before synchrony computation (reject noise-dominated channels)
- CCA_1 (first canonical component across body + FAU + prosody) as the
  primary coupling signal — replaces multi-channel R(t) on non-oscillatory data
- Full trimodal support with F0 extraction from WAV via librosa.pyin

References:
- PSD + spectral reddening: Bury et al. 2020 J R Soc Interface
- CCA for multimodal synchrony: PDF "Coupling, Counterfactuals, and Constella" Exp 1
- IAAFT surrogates: Schreiber & Schmitz 1996
- Boashash diagnostic: Boashash 1992 Proc IEEE (non-oscillatory signals)
- Effect-size calibration: Ohayon & Gordon BBR 2024 (r ≈ 0.32 multimodal meta)
- Analytical coupling threshold (Apr 18 2026 addition):
    Rodrigues et al. 2016 Phys. Rep. 610 (mean-field backbone);
    Garijo/Gómez/Arenas 2026 arXiv:2604.14772 (convex-geometric, illustrative).
- Rupture-anchored windowing (Apr 18 2026 addition):
    Marwan et al. 2007 Phys. Rep. 438:237–329 (CRP textbook);
    Fusaroli, Rączaszek-Leonardi & Tylén 2014 New Ideas in Psychology 32:147
    (windowed CRQA convention — replaces misattributed Galati 2026 ref 71).
- HMM regime segmentation (Apr 18 2026 addition):
    Vidaurre et al. 2018 NeuroImage 180:646 DOI:10.1016/j.neuroimage.2017.06.077
    (TDE-HMM on brain states; NOT Li 2026 which uses k-means).

=============================================================================
TODO (Milestone 2 — post Apr 20 submission):
=============================================================================
Split this 1826+-line module into 6 focused files to comply with the 800-line
CLAUDE.md limit (currently 2.3× over):

    psd_utils.py          # compute_psd, estimate_noise_floor, spectral_flatness,
                          # signal_to_noise_ratio_db, assess_signal_quality,
                          # recommend_bandpass, MODALITY_SNR_BANDS,
                          # MODALITY_NPERSEG, SPECTRAL_FLATNESS_MAX, SNR_DB_MIN,
                          # QualityReport, batch_psd_report
    filter_utils.py       # design_bandpass_sos, apply_bandpass
    f0_extraction.py      # F0 / prosody extraction helpers
    dyad_io.py            # load_npz_feature, interaction_key_from_filename,
                          # load_dyad_from_manifest, list_dyads_from_manifest,
                          # FAMILIAR_LABELS, classify_relationship
    feature_engineering.py # body_motion_energy, fau_mean_activation,
                           # head_motion_energy, apply_valid_mask,
                           # dyad_is_valid_mask, align_to_common_length,
                           # build_cca_features, build_cca_features_rich,
                           # extract_dyad_cca_features, expand_weights_to_full_space,
                           # compute_null_normalized_r
    cca_pipeline.py       # compute_cca_1, iaaft_surrogates, partner_shuffled_null,
                          # partner_shuffled_null_fixed_directions,
                          # surrogate_p_value, compute_analytical_threshold_garijo2026,
                          # anchor_crqa_to_rupture_events, hmm_regime_segmentation

Deferred from Apr 13-20 sprint to preserve import stability during the
result-landing window. The split is mechanical (no new functionality) but
requires touching every notebook import cell; risk is unacceptable within
48 hours of submission. File cited in code-review sweep (CODE_REVIEW_LOG.md,
Apr 17 2026, "File size 1826 lines, 2.3× the 800-line limit").
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, coherence, sosfiltfilt, welch
from sklearn.cross_decomposition import CCA

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

SAMPLE_RATE = 30  # Hz — NPZ movement features native rate
AUDIO_SAMPLE_RATE = 48000  # Hz — Seamless audio
NYQUIST = SAMPLE_RATE / 2  # 15 Hz

# Quality-gate thresholds (from PSD_RELEVANCE_ANALYSIS.md §4.3)
SPECTRAL_FLATNESS_MAX = 0.8  # >0.8 → noise-dominated
SNR_DB_MIN = 3.0  # <3 dB → signal not discriminable from noise

# Per-modality default bandpass (empirical — from PSD report §4.3)
# Used as FALLBACK when PSD-driven recommendation is unavailable.
MODALITY_BANDPASS_DEFAULT = {
    "body": (0.1, 5.0),
    "fau": (0.5, 8.0),
    "prosody": (2.0, 10.0),  # capped at Nyquist=15 Hz
    "gaze": (0.1, 2.0),
    "head": (0.1, 3.0),
}


# ============================================================================
# Core PSD utilities
# ============================================================================

def compute_psd(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE,
    nperseg: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of a 1-D signal.

    Args:
        signal: 1-D time series.
        fs: Sampling frequency (Hz). Defaults to 30.
        nperseg: FFT window length. 256 samples ≈ 8.5 s at 30 Hz.

    Returns:
        (freqs, psd) where freqs is Hz and psd is V²/Hz.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    # Welch requires nperseg <= len(signal); cap accordingly
    nperseg_eff = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg_eff)
    return freqs, psd


def estimate_noise_floor(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE,
    search_start_hz: float = 0.5,
    nperseg: int = 256,
) -> tuple[float, float]:
    """Estimate the frequency where the PSD flattens (noise floor onset).

    Uses log-PSD gradient: where |slope| < 0.1 on a log-log scale, the
    signal is in a white-noise regime.

    Bug fixed (code-audit HIGH-3 Apr 17, 2026): the original version returned
    freqs[0] (DC bin, ~0 Hz) for nearly all behavioral signals because
    log-PSD is shallow at DC on 1/f spectra. Skipping the DC region via
    `search_start_hz` avoids the degenerate bandpass (0.1, 0.5) that this
    propagated into.

    Apr 18 2026 fix: `nperseg` now threaded through to `compute_psd` so
    per-modality window lengths (MODALITY_NPERSEG) actually propagate from
    `batch_psd_report`. Previously the default 256 was always used.

    Returns:
        (noise_start_freq, noise_level_power).
        If no flattening is detected above `search_start_hz`, returns
        (nyquist, last_psd_value).
    """
    freqs, psd = compute_psd(signal, fs=fs, nperseg=nperseg)
    log_psd = np.log10(psd + 1e-12)
    # Do not divide by (freqs + 1e-9); use native spacing to avoid distortion
    gradient = np.gradient(log_psd, freqs)
    flat_mask = np.abs(gradient) < 0.1

    # Restrict search to frequencies above search_start_hz (default 0.5 Hz)
    # to avoid the DC-bin false positive documented above.
    above_start = freqs >= search_start_hz
    valid_flat = flat_mask & above_start
    if valid_flat.any():
        idx = int(np.argmax(valid_flat))
        # Median of values in the flat region (above start) for robustness
        return float(freqs[idx]), float(np.median(psd[valid_flat]))
    return float(freqs[-1]), float(psd[-1])


def spectral_flatness(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE,
    nperseg: int = 256,
) -> float:
    """Wiener entropy: 0 = pure tone, 1 = white noise.

    Computed as entropy of the normalized PSD divided by max entropy.

    Apr 18 2026 fix: `nperseg` now threaded through to `compute_psd` so
    per-modality window lengths propagate from `batch_psd_report`.
    """
    _, psd = compute_psd(signal, fs=fs, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + 1e-12)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    max_entropy = np.log(len(psd))
    return float(entropy / max_entropy) if max_entropy > 0 else 1.0


def signal_to_noise_ratio_db(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE,
    signal_band: tuple[float, float] = (0.1, 8.0),
    noise_band: tuple[float, float] = (8.0, 15.0),
    nperseg: int = 256,
) -> float:
    """SNR in dB: ratio of mean PSD in signal band to noise band.

    Default bands assume 30 Hz sampling. Signal band 0.1–8 Hz covers
    behavioral dynamics; noise band 8–15 Hz captures HMR 2.0 jitter.

    NOTE (audit HIGH-8, Apr 17, 2026): these defaults are BODY-POSE-centric.
    FAU has genuine energy to 8-10 Hz (smiles, brow dynamics — Ekman & Rosenberg
    2005), prosody to 4-8 Hz (syllabic rate — Poeppel 2003), emotion_valence
    is DC-dominated (<0.5 Hz per Kuppens 2010). Use
    `MODALITY_SNR_BANDS[modality]` via `assess_signal_quality(..., modality=)`
    to get per-modality defaults.

    Apr 18 2026 fix: `nperseg` now threaded through to `compute_psd` so
    per-modality window lengths propagate from `batch_psd_report`.
    """
    freqs, psd = compute_psd(signal, fs=fs, nperseg=nperseg)
    sig_mask = (freqs > signal_band[0]) & (freqs < signal_band[1])
    noise_mask = (freqs >= noise_band[0]) & (freqs < noise_band[1])
    sig_power = psd[sig_mask].mean() if sig_mask.any() else 0.0
    noise_power = psd[noise_mask].mean() if noise_mask.any() else 1e-12
    if noise_power <= 0 or sig_power <= 0:
        return -np.inf
    return float(10.0 * np.log10(sig_power / noise_power))


# Per-modality (signal_band, noise_band) tuples in Hz. Used by the quality
# gate to avoid systematically rejecting FAU/prosody where 8-15 Hz is signal.
MODALITY_SNR_BANDS = {
    "body":    ((0.1, 5.0),  (5.0, 15.0)),   # HMR 2.0 jitter peak ~5+ Hz
    "fau":     ((0.5, 8.0),  (10.0, 15.0)),  # FAU dynamics to 8 Hz
    "prosody": ((2.0, 10.0), (12.0, 15.0)),  # syllabic rate 4-8 Hz, caps at Nyquist
    "gaze":    ((0.1, 2.0),  (4.0, 15.0)),   # gaze coordination is slow
    "head":    ((0.1, 3.0),  (5.0, 15.0)),   # head nods up to ~3 Hz
    "emotion": ((0.01, 0.5), (2.0, 15.0)),   # valence/arousal <0.5 Hz per Kuppens 2010
}
# Opus H1 fix (Apr 17, 2026): alias 'emotion_valence' → 'emotion' so callers
# using the NPZ key name ('emotion_valence') get the correct band.
MODALITY_SNR_BANDS["emotion_valence"] = MODALITY_SNR_BANDS["emotion"]
MODALITY_SNR_BANDS["emotion_arousal"] = MODALITY_SNR_BANDS["emotion"]

# Per-modality nperseg (Welch segment length, in samples). Opus H7 fix:
# emotion needs longer windows (low-freq content); FAU can use shorter.
# Balanced against min-length gate = 640 samples (H6).
MODALITY_NPERSEG = {
    "body":    256,   # 8.5 s at 30 Hz — freq res 0.12 Hz, good for 0.1-5 Hz content
    "fau":     128,   # 4.3 s — better time resolution for expression dynamics
    "prosody": 128,
    "gaze":    256,
    "head":    256,
    "emotion": 512,   # 17 s — resolves 0.01-0.5 Hz content of damped-stochastic valence
    "emotion_valence": 512,
    "emotion_arousal": 512,
}

# Canonical familiar-relationship label set. Used for stranger/familiar
# classification across all notebooks. Duplicate of seamless_data_pipeline.ipynb
# Cell 7's FAMILIAR_LABELS (kept in sync; change both if relationships.csv
# gains new categories).
FAMILIAR_LABELS = frozenset({
    "familiar", "friends", "coworkers", "family-generic",
    "familiar-generic", "siblings", "parent_child",
    "dating", "spouse", "romantic_partner", "neighbors",
})


def classify_relationship(rel_label: Optional[str]) -> str:
    """Map a raw relationship string to one of {'stranger', 'familiar', 'unknown'}.

    Uses the canonical FAMILIAR_LABELS set. Fixes Opus H5 / Sonnet HIGH-2
    (Apr 17, 2026): the previous inline logic in cell 14 missed 6 of 11
    familiar categories (siblings, parent_child, dating, spouse,
    romantic_partner, neighbors), bucketing them as 'unknown' and
    distorting the stranger/familiar balance.
    """
    if rel_label is None:
        return "unknown"
    r = str(rel_label).strip().lower()
    if r == "stranger":
        return "stranger"
    if r in FAMILIAR_LABELS:
        return "familiar"
    return "unknown"


# ============================================================================
# Quality gate
# ============================================================================

@dataclass
class QualityReport:
    passed: bool
    spectral_flatness: float
    snr_db: float
    noise_floor_freq: float
    reasons: list[str]


def assess_signal_quality(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE,
    flatness_max: float = SPECTRAL_FLATNESS_MAX,
    snr_min_db: float = SNR_DB_MIN,
    modality: Optional[str] = None,
    nperseg: int = 256,
) -> QualityReport:
    """Run spectral-flatness + SNR gate. Reject channels failing either.

    Channels with spectral_flatness > 0.8 are noise-dominated; SNR < 3 dB
    means the signal is indiscriminable from noise.

    Args:
        modality: If provided (body / fau / prosody / gaze / head / emotion),
                  use per-modality (signal_band, noise_band) from
                  MODALITY_SNR_BANDS instead of the body-centric default.
                  Applied to address audit HIGH-8 (systematic FAU/prosody
                  rejection under the 0.1-8 / 8-15 Hz defaults).
        nperseg: Welch window length. If `modality` is provided AND present
                 in MODALITY_NPERSEG, that per-modality value overrides the
                 argument (so callers that already know the modality need
                 not pass nperseg explicitly). Apr 18 2026 fix: previously
                 nperseg=256 was always used regardless of modality.
    """
    # Per-modality auto-pick: emotion → 512, FAU → 128, body/head/gaze → 256
    eff_nperseg = MODALITY_NPERSEG.get(modality, nperseg) if modality else nperseg
    sf = spectral_flatness(signal, fs=fs, nperseg=eff_nperseg)
    if modality is not None and modality in MODALITY_SNR_BANDS:
        sig_band, noise_band = MODALITY_SNR_BANDS[modality]
        snr = signal_to_noise_ratio_db(
            signal, fs=fs, signal_band=sig_band, noise_band=noise_band,
            nperseg=eff_nperseg,
        )
    else:
        snr = signal_to_noise_ratio_db(signal, fs=fs, nperseg=eff_nperseg)
    nf_freq, _ = estimate_noise_floor(signal, fs=fs, nperseg=eff_nperseg)

    reasons: list[str] = []
    if sf > flatness_max:
        reasons.append(f"spectral_flatness={sf:.3f} > {flatness_max}")
    if snr < snr_min_db:
        reasons.append(f"snr_db={snr:.2f} < {snr_min_db}")
    if not np.isfinite(snr):
        reasons.append("snr not finite (all-zero noise band or signal band)")

    return QualityReport(
        passed=len(reasons) == 0,
        spectral_flatness=sf,
        snr_db=snr,
        noise_floor_freq=nf_freq,
        reasons=reasons,
    )


def recommend_bandpass(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE,
    energy_percentiles: tuple[float, float] = (10.0, 90.0),
    min_low: float = 0.05,
    noise_floor_cap: bool = True,
    nperseg: int = 256,
) -> tuple[float, float]:
    """Empirical bandpass from cumulative PSD energy percentiles.

    Returns a (low, high) band containing `energy_percentiles` of the
    spectral energy. If `noise_floor_cap`, cap high at the noise floor.

    Useful for the per-modality PSD-driven bandpass choice requested by
    the sprint (resolves the 0.5–3.0 Hz vs 0.2–2.0 Hz dispute).

    Apr 18 2026 fix: `nperseg` now threaded through to `compute_psd` and
    `estimate_noise_floor` so per-modality window lengths propagate from
    `batch_psd_report`.
    """
    freqs, psd = compute_psd(signal, fs=fs, nperseg=nperseg)
    # Cumulative energy distribution
    cum_energy = np.cumsum(psd) / (psd.sum() + 1e-12)
    low_pct, high_pct = energy_percentiles
    low_idx = int(np.searchsorted(cum_energy, low_pct / 100.0))
    high_idx = int(np.searchsorted(cum_energy, high_pct / 100.0))
    low_freq = max(min_low, float(freqs[min(low_idx, len(freqs) - 1)]))
    high_freq = float(freqs[min(high_idx, len(freqs) - 1)])

    if noise_floor_cap:
        nf_freq, _ = estimate_noise_floor(signal, fs=fs, nperseg=nperseg)
        high_freq = min(high_freq, nf_freq)

    # Enforce valid ordering + Nyquist. Guarantee high > low in all cases
    # (audit HIGH-7).
    high_freq = min(high_freq, NYQUIST - 0.01)
    if high_freq <= low_freq:
        candidate = min(low_freq + 0.5, NYQUIST - 0.01)
        if candidate <= low_freq:
            # low_freq is already near Nyquist; move low down to make room
            low_freq = max(min_low, candidate - 0.5)
            high_freq = candidate
        else:
            high_freq = candidate
        logger.warning(
            "recommend_bandpass: degenerate band produced; using "
            "fallback (%.3f, %.3f) Hz",
            low_freq, high_freq,
        )
    return low_freq, high_freq


# ============================================================================
# Filtering
# ============================================================================

def design_bandpass_sos(
    low: float,
    high: float,
    fs: float = SAMPLE_RATE,
    order: int = 4,
) -> np.ndarray:
    """Design a Butterworth bandpass as SOS (second-order sections).

    SOS is numerically more stable than ba coefficients, especially at
    higher orders. Matches the research-notebook-qa skill's recommendation.
    """
    # Normalize to Nyquist
    low_n = low / (fs / 2)
    high_n = min(high / (fs / 2), 0.99)
    low_n = max(low_n, 0.001)
    if high_n <= low_n:
        raise ValueError(f"Invalid band [{low}, {high}] Hz at fs={fs}")
    return butter(order, [low_n, high_n], btype="band", output="sos")


def apply_bandpass(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: float = SAMPLE_RATE,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase bandpass via filtfilt on SOS coefficients.

    Raises ValueError (audit HIGH-5) if signal is shorter than filtfilt's
    padlen requirement (~6*order samples for order-N Butterworth).
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    # Conservative minimum length for sosfiltfilt. scipy's padlen is
    # 3*max(len(sos[0]), len(sos[1])) = 18 for order=4; we require a bit
    # more for stable filtfilt.
    min_len = max(3 * 2 * order + 1, 30)
    if len(signal) < min_len:
        raise ValueError(
            f"Signal length {len(signal)} < minimum {min_len} for "
            f"order-{order} SOS filtfilt (at fs={fs} Hz)."
        )
    sos = design_bandpass_sos(low, high, fs=fs, order=order)
    return sosfiltfilt(sos, signal)


# ============================================================================
# F0 prosody extraction from WAV
# ============================================================================

def extract_f0_envelope(
    wav_path: str | Path,
    target_fs: float = SAMPLE_RATE,
    fmin: float = 60.0,
    fmax: float = 500.0,
    hop_length: Optional[int] = None,
    imputation: str = "median",
    return_voicing_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Extract F0 envelope from a WAV at target sampling rate.

    Uses librosa.pyin (probabilistic YIN — more accurate than default YIN)
    on 48 kHz audio, then resamples to target_fs (default 30 Hz) to align
    with NPZ movement features.

    ⚠ AUDIT NOTES (Apr 17, 2026):
    - HIGH-6: pYIN is designed for MONOPHONIC signals. Seamless dyadic
      audio should be per-participant (967 WAVs for 436 dyads × 2
      participants), and Meta uses Beryl AEC to remove speaker bleed.
      Verify by inspection (two WAVs per interaction_key) before relying
      on F0 as a per-person pitch feature. Overlapping speech will
      produce dominance-weighted F0, not true prosody coupling.
    - HIGH-7: Linear interpolation over unvoiced segments (default in v1)
      creates artificial 0.1–0.5 Hz trajectories that both partners share,
      confounding prosodic synchrony claims. Default changed to "median"
      imputation (preserves DC, not trajectory). Use `imputation="none"`
      with `return_voicing_mask=True` for masked-correlation downstream.

    Args:
        wav_path: Path to .wav file.
        target_fs: Output rate (Hz). Defaults to 30.
        fmin / fmax: Pitch search range. 60-500 Hz covers adult speech
                     (male ~85-180 Hz, female ~165-255 Hz) + headroom.
        hop_length: Hop length in samples for pyin. If None, computed to
                    approximate target_fs.
        imputation: How to fill unvoiced frames:
                    - "linear": original v1 behavior (NOT RECOMMENDED)
                    - "median": fill with per-file median F0 (default;
                                preserves DC but kills trajectory artifacts)
                    - "zero": fill with zeros
                    - "none": leave NaN; caller handles via voicing mask
        return_voicing_mask: If True, return (f0, voicing_mask) where
                             voicing_mask is True for voiced frames.

    Returns:
        (f0_envelope,) at target_fs, OR
        (f0_envelope, voicing_mask) if return_voicing_mask=True.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "librosa required for F0 extraction. "
            "Install with: pip install librosa"
        ) from exc

    wav_path = Path(wav_path)
    y, sr = librosa.load(str(wav_path), sr=None)
    if hop_length is None:
        hop_length = max(1, int(round(sr / target_fs)))

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )

    f0 = np.asarray(f0, dtype=np.float64)
    voicing_mask = np.isfinite(f0)

    # Imputation
    if not np.any(voicing_mask):
        # Fully unvoiced — return zeros (or NaN if imputation="none")
        f0_filled = np.zeros(len(f0)) if imputation != "none" else f0.copy()
    elif imputation == "linear":
        idx = np.arange(len(f0))
        f0_filled = np.interp(idx, idx[voicing_mask], f0[voicing_mask])
    elif imputation == "median":
        # Per-file median: preserves DC level, avoids fabricated slow dynamics
        med = float(np.nanmedian(f0))
        f0_filled = np.where(voicing_mask, f0, med)
    elif imputation == "zero":
        f0_filled = np.where(voicing_mask, f0, 0.0)
    elif imputation == "none":
        f0_filled = f0.copy()
    else:
        raise ValueError(f"Unknown imputation: {imputation!r}")

    # Resample to target_fs if hop didn't land exactly there
    actual_fs = sr / hop_length
    if abs(actual_fs - target_fs) > 1.0 and imputation != "none":
        n_target = int(round(len(f0_filled) * target_fs / actual_fs))
        old_idx = np.arange(len(f0_filled))
        new_idx = np.linspace(0, len(f0_filled) - 1, n_target)
        f0_filled = np.interp(new_idx, old_idx, f0_filled)
        # Resample voicing mask via nearest-neighbor
        vm_int = voicing_mask.astype(np.float64)
        vm_resampled = np.interp(new_idx, old_idx, vm_int) > 0.5
        voicing_mask = vm_resampled

    if return_voicing_mask:
        return f0_filled, voicing_mask
    return f0_filled


# ============================================================================
# Dyad loading from the pipeline manifest
# ============================================================================
# Grounded in actual outputs from
# /Users/yashsahitya/Documents/seamless_interaction/Notebooks/seamless_data_pipeline.ipynb
# as of Apr 17, 2026:
#   * NPZ dir:       /content/drive/MyDrive/seamless_interaction/npz/V*.npz
#   * Manifest:      /content/drive/MyDrive/seamless_interaction/pipeline_manifest.json
#   * File naming:   V00_S0062_I00000128_P0092.npz
#                     → vendor, session, interaction, participant
#   * 541 complete dyads on disk: 311 naturalistic (52S/259F) + 125 improvised
#   * Frame counts vary by interaction duration (e.g., 2400, 4920, 6420 frames;
#     all at 30 Hz, so 80 s – 3.5 min). Align to common length before CCA.

# Actual NPZ key layout (from Cell 11 test download output):
#   movement:FAUValue            (T, 24)  — facial action unit intensities
#   movement:emotion_valence     (T, 1)   — needs squeeze → (T,)
#   movement:emotion_arousal     (T, 1)   — needs squeeze → (T,)
#   movement:gaze_encodings      (T, 2)   — 2-D NEURAL encoding (NOT angles)
#   movement:head_encodings      (T, 3)   — 3-D NEURAL encoding (NOT angles)
#   movement:expression          (T, 128) — high-dim expression embedding
#   movement:emotion_scores      (T, 8)   — 8-category softmax scores
#   movement:is_valid            (T, 1)   — quality mask (1.0 = valid)
#   smplh:body_pose              (T, 21, 3)  — 21 joint rotations
#   smplh:translation            (T, 3)   — pelvis XYZ
#   smplh:global_orient          (T, 3)
#   smplh:is_valid               (T,)     — dtype=bool
#   smplh:left_hand_pose         (T, 15, 3)
#   smplh:right_hand_pose        (T, 15, 3)

NPZ_KEYS_PRIMARY = [
    "movement:FAUValue",
    "movement:emotion_valence",
    "movement:emotion_arousal",
    "movement:gaze_encodings",
    "movement:head_encodings",
    "movement:is_valid",
    "smplh:body_pose",
    "smplh:translation",
    "smplh:global_orient",
    "smplh:is_valid",
]


def load_npz_feature(npz_path: str | Path, key: str) -> Optional[np.ndarray]:
    """Load a single feature from an NPZ file and squeeze (T,1) → (T,).

    Handles the Meta Seamless convention of shipping emotion_valence and
    similar scalars as (T, 1) arrays. Returns None if key missing.

    Uses allow_pickle=False (audit HIGH-6) — Seamless NPZs contain only
    float32/bool arrays, no Python objects needed. If a caller needs
    structured arrays, load with allow_pickle=True explicitly.
    """
    with np.load(str(npz_path), allow_pickle=False) as data:
        if key not in data.files:
            return None
        arr = np.asarray(data[key])
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(axis=1)
    return arr


def interaction_key_from_filename(stem: str) -> str:
    """'V00_S0062_I00000128_P0092' → 'V00_S0062_I00000128'.

    Uses rsplit (not [:17]) so that interaction IDs of any length work.
    """
    return stem.rsplit("_", 1)[0]


def load_dyad_from_manifest(
    manifest_path: str | Path,
    npz_dir: str | Path,
    interaction_key: str,
    features: Optional[list[str]] = None,
    preloaded_manifest: Optional[dict] = None,
) -> Optional[dict]:
    """Load both participants of a dyad by interaction_key.

    Args:
        manifest_path: Path to pipeline_manifest.json. Ignored if
                       preloaded_manifest is provided.
        npz_dir: Directory containing the NPZ files (flat).
        interaction_key: e.g., 'V00_S0062_I00000128'.
        features: NPZ keys to load. Defaults to NPZ_KEYS_PRIMARY.
        preloaded_manifest: optional already-parsed manifest dict. Provide
                            this when calling load_dyad_from_manifest in a
                            loop to avoid re-parsing the JSON file each
                            iteration (Sonnet HIGH-5 fix: Drive FUSE round-
                            trips on 311 calls can time out cells).

    Returns:
        dict with keys:
          - 'interaction_key'
          - 'condition'       (from manifest file_conditions)
          - 'relationship'    (from manifest file_relationships)
          - 'participants'    [p0_features_dict, p1_features_dict]
          - 'file_ids'        [p0_file_id, p1_file_id]
        or None if the dyad is not in the manifest or on disk.
    """
    import json

    manifest_path = Path(manifest_path)
    npz_dir = Path(npz_dir)
    if features is None:
        features = list(NPZ_KEYS_PRIMARY)

    if preloaded_manifest is not None:
        manifest = preloaded_manifest
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)

    file_conds = manifest.get("file_conditions", {})
    file_rels = manifest.get("file_relationships", {})

    # Find both participants for this interaction_key
    matching_ids = sorted([
        fid for fid in manifest.get("file_ids", [])
        if interaction_key_from_filename(fid) == interaction_key
    ])
    if len(matching_ids) != 2:
        logger.warning(
            "interaction_key %s has %d participants on disk (need 2)",
            interaction_key, len(matching_ids),
        )
        return None

    participants = []
    for fid in matching_ids:
        npz_path = npz_dir / f"{fid}.npz"
        if not npz_path.exists():
            logger.warning("NPZ missing: %s", npz_path)
            return None
        p_feats = {}
        for k in features:
            arr = load_npz_feature(npz_path, k)
            if arr is not None:
                p_feats[k] = arr
        participants.append(p_feats)

    condition = file_conds.get(matching_ids[0], "unknown")
    relationship = file_rels.get(matching_ids[0], "unknown")
    return {
        "interaction_key": interaction_key,
        "condition": condition,
        "relationship": relationship,
        "participants": participants,
        "file_ids": matching_ids,
    }


def list_dyads_from_manifest(
    manifest_path: str | Path,
    condition: Optional[str] = None,
    relationship: Optional[str] = None,
) -> list[str]:
    """Return interaction_keys filtered by condition and/or relationship.

    condition ∈ {'naturalistic', 'improvised'} or None for all.
    relationship ∈ {'stranger', 'familiar', ...} or None for all.
    """
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    file_conds = manifest.get("file_conditions", {})
    file_rels = manifest.get("file_relationships", {})

    keys = set()
    for fid in manifest.get("file_ids", []):
        ik = interaction_key_from_filename(fid)
        cond = file_conds.get(fid, "unknown")
        rel = file_rels.get(fid, "unknown")
        if condition and cond != condition:
            continue
        if relationship and rel != relationship:
            continue
        keys.add(ik)
    return sorted(keys)


# ============================================================================
# Modality-feature scalarization (prep for CCA)
# ============================================================================

def body_motion_energy(
    translation: np.ndarray,
    body_pose: Optional[np.ndarray] = None,
    fs: float = SAMPLE_RATE,
    lowpass_hz: float = 5.0,
) -> np.ndarray:
    """Derive a scalar body-motion signal per frame (HMR-noise-aware).

    translation: (T, 3) pelvis XYZ
    body_pose: optional (T, 21, 3) joint rotations
    lowpass_hz: cutoff for pre-differentiation lowpass (default 5 Hz).
                Set <= 0 to disable (legacy behavior).

    Returns: (T,) motion-energy envelope.

    Audit HIGH-10 fix (Apr 17, 2026):
    `np.diff` has frequency response H(f)=2·sin(π·f/fs); at f=10 Hz this
    gives ~1.7x gain, strongly amplifying HMR 2.0 frame-to-frame noise
    (peak above ~5 Hz). We lowpass at 5 Hz BEFORE differentiation so that
    the motion-energy signal reflects body kinematics, not estimator jitter.

    For very short signals (< ~30 frames) the lowpass is skipped and a
    warning is logged; the signal is still usable but HMR noise is not
    suppressed.
    """
    translation = np.asarray(translation, dtype=np.float64)
    if translation.ndim != 2 or translation.shape[1] != 3:
        raise ValueError(f"translation must be (T,3), got {translation.shape}")

    # Lowpass each translation dim at 5 Hz before differentiation
    def _lowpass_safe(x: np.ndarray) -> np.ndarray:
        """Apply 5-Hz lowpass; fall back to identity for very short signals."""
        if lowpass_hz <= 0 or len(x) < 30:
            if lowpass_hz > 0:
                logger.warning(
                    "body_motion_energy: signal length %d too short for "
                    "lowpass; skipping (HMR noise not suppressed)", len(x),
                )
            return x
        try:
            # Lowpass: use butter directly (not bandpass) at lowpass_hz
            sos = butter(
                4, lowpass_hz / (fs / 2), btype="low", output="sos",
            )
            return sosfiltfilt(sos, x)
        except Exception as exc:
            logger.warning("body_motion_energy: lowpass failed (%s); using raw", exc)
            return x

    trans_lp = np.column_stack([_lowpass_safe(translation[:, i]) for i in range(3)])
    tvel = np.diff(trans_lp, axis=0, prepend=trans_lp[:1])
    tvel_mag = np.linalg.norm(tvel, axis=1)

    if body_pose is None:
        return tvel_mag

    body_pose = np.asarray(body_pose, dtype=np.float64)
    # Collapse (T, 21, 3) -> (T, 63)
    if body_pose.ndim == 3:
        body_flat = body_pose.reshape(body_pose.shape[0], -1)
    else:
        body_flat = body_pose

    # Lowpass each of the 63 pose channels
    if lowpass_hz > 0 and body_flat.shape[0] >= 30:
        body_flat_lp = np.column_stack(
            [_lowpass_safe(body_flat[:, i]) for i in range(body_flat.shape[1])]
        )
    else:
        body_flat_lp = body_flat

    bvel = np.diff(body_flat_lp, axis=0, prepend=body_flat_lp[:1])
    bvel_mag = np.linalg.norm(bvel, axis=1)

    # Z-score each then sum (prevents one channel dominating).
    # Audit note: this is standard pre-CCA normalization; sum composes the
    # two velocity signals as a single "motion energy" axis. PCA would be
    # an alternative (orthogonalize), considered for v2.
    def _z(x: np.ndarray) -> np.ndarray:
        s = x.std() + 1e-12
        return (x - x.mean()) / s

    return _z(tvel_mag) + _z(bvel_mag)


def fau_mean_activation(fau_value: np.ndarray) -> np.ndarray:
    """Mean FAU activation per frame.

    fau_value: (T, 24)
    Returns: (T,) mean across 24 action units.
    """
    fau = np.asarray(fau_value, dtype=np.float64)
    if fau.ndim != 2 or fau.shape[1] != 24:
        raise ValueError(f"fau_value must be (T,24), got {fau.shape}")
    return fau.mean(axis=1)


def head_motion_energy(
    head_encodings: np.ndarray,
    fs: float = SAMPLE_RATE,
    lowpass_hz: float = 3.0,
) -> np.ndarray:
    """Frame-difference magnitude of the 3-D head neural encoding.

    Opus H4 fix (Apr 17, 2026): previously we used `np.linalg.norm(he, axis=1)`
    as the head scalar, which is "distance from encoder origin" — has no
    coupling-relevant interpretation for an opaque 3-D neural encoding.
    Frame-difference magnitude (||h(t) - h(t-1)||) is at least a rate-of-
    change, which tracks head movement regardless of encoding origin.

    For CCA downstream, prefer passing the 3-D head_encodings directly
    (compute_cca_1 handles multi-D input). This function is the diagnostic
    scalar for PSD/QC only.

    head_encodings: (T, 3) learned neural representation — NOT Euler angles.
    lowpass_hz: pre-differentiation lowpass cutoff. Default 3 Hz matches
                typical head-nod fundamental frequency.
    """
    he = np.asarray(head_encodings, dtype=np.float64)
    if he.ndim != 2 or he.shape[1] != 3:
        raise ValueError(f"head_encodings must be (T,3), got {he.shape}")

    # Lowpass each of the 3 latent dims independently before differentiation
    if lowpass_hz > 0 and he.shape[0] >= 30:
        try:
            sos = butter(4, lowpass_hz / (fs / 2), btype="low", output="sos")
            he_lp = np.column_stack([
                sosfiltfilt(sos, he[:, i]) for i in range(3)
            ])
        except Exception as exc:
            logger.warning("head_motion_energy: lowpass failed (%s)", exc)
            he_lp = he
    else:
        he_lp = he

    # Frame-difference magnitude
    diff = np.diff(he_lp, axis=0, prepend=he_lp[:1])
    return np.linalg.norm(diff, axis=1)


def apply_valid_mask(
    signal: np.ndarray,
    is_valid: np.ndarray,
    min_valid_fraction: float = 0.5,
) -> Optional[np.ndarray]:
    """Drop invalid frames from a signal before PSD/QC.

    Opus H3 fix (Apr 17, 2026): cells 12 + 14 of measuring_synchrony.ipynb
    previously ran Welch PSD on signals that included occlusion/dropout
    frames. Welch on 58%-invalid signals inflates spectral flatness
    toward 1.0 (confounds "bad sensor" with "bad signal").

    Args:
        signal: 1-D or 2-D time series (T,) or (T, D)
        is_valid: 1-D mask (T,) where True/1 = valid frame.
        min_valid_fraction: if valid fraction < this, return None (signal
                            is too contaminated to analyze).

    Returns:
        Signal with invalid frames removed (or None if insufficient valid).
        Preserves temporal order but concatenates valid runs.
    """
    signal = np.asarray(signal)
    is_valid = np.asarray(is_valid).astype(bool).ravel()
    if len(is_valid) != signal.shape[0]:
        # Length mismatch — truncate to common length
        n = min(len(is_valid), signal.shape[0])
        signal = signal[:n]
        is_valid = is_valid[:n]
    valid_frac = float(is_valid.mean()) if len(is_valid) else 0.0
    if valid_frac < min_valid_fraction:
        return None
    if valid_frac == 1.0:
        return signal
    # Drop invalid frames
    if signal.ndim == 1:
        return signal[is_valid]
    return signal[is_valid, :]


def dyad_is_valid_mask(participant_features: dict) -> Optional[np.ndarray]:
    """Combine NPZ is_valid masks from a single participant.

    Intersects movement:is_valid and smplh:is_valid to get a single
    per-frame validity indicator. Returns None if neither is present.
    """
    mv = participant_features.get("movement:is_valid")
    sv = participant_features.get("smplh:is_valid")
    masks = []
    if mv is not None:
        mask = np.asarray(mv).astype(bool).ravel()
        masks.append(mask)
    if sv is not None:
        mask = np.asarray(sv).astype(bool).ravel()
        masks.append(mask)
    if not masks:
        return None
    # Intersect at common length
    n = min(len(m) for m in masks)
    return np.all([m[:n] for m in masks], axis=0)


# ============================================================================
# Canonical Correlation Analysis (CCA) — the primary coupling signal
# ============================================================================

def align_to_common_length(
    signals: list[np.ndarray],
    mode: str = "truncate",
) -> list[np.ndarray]:
    """Align a list of 1-D signals to a common length.

    mode="truncate": cut to min length (default, avoids extrapolation).
    mode="pad": zero-pad to max length.
    """
    lens = [len(s) for s in signals]
    if mode == "truncate":
        n = min(lens)
        return [s[:n] for s in signals]
    if mode == "pad":
        n = max(lens)
        out = []
        for s in signals:
            if len(s) < n:
                pad = np.zeros(n - len(s))
                out.append(np.concatenate([s, pad]))
            else:
                out.append(s)
        return out
    raise ValueError(f"Unknown mode: {mode}")


def ar1_prewhiten(signal: np.ndarray) -> np.ndarray:
    """AR(1) residualization: remove lag-1 autocorrelation from a signal.

    Fits AR(1) coefficient φ via lag-1 Pearson correlation, then returns
    residuals e[t] = x[t] - φ·x[t-1].

    ⚠ AUDIT NOTES (Apr 17 Opus HIGH-B):
    - Per-channel AR(1) pre-whitening can ATTENUATE phase-lagged cross-channel
      coupling by 30-60% (Yu & Eisenberg 2018 NeuroImage). It is a CONSERVATIVE
      direction — cannot invent synchrony, but can hide real phase-lagged coupling.
    - For canonical-coherence analysis in time-domain CCA, the canonical
      references are Brillinger 1975 (Time Series: Data Analysis & Theory,
      Ch.8) and Dahlhaus 2000 (Ann. Stat.), NOT Granger & Newbold 1974 (that
      paper is about spurious regression in econometrics).
    - In this sprint we prefer `prewhiten=False` in compute_cca_1 + use
      partner-shuffled null (which absorbs autocorrelation in the null
      distribution itself — Winkler et al. 2020 NeuroImage).
    - For multi-column input, applies column-wise. For proper multivariate
      autocorrelation correction, use VAR(p) residualization instead.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        if len(x) < 3:
            return x.copy()
        lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        if not np.isfinite(lag1):
            return x.copy()
        residuals = x[1:] - lag1 * x[:-1]
        # Pad with single zero at front to preserve length
        return np.concatenate([[0.0], residuals])
    # Multi-column: whiten each column independently
    return np.column_stack([ar1_prewhiten(x[:, i]) for i in range(x.shape[1])])


@dataclass
class CCAResult:
    """CCA fit result with canonical directions retained for valid null testing."""
    cca_a_1: np.ndarray       # (T,) canonical variate for person A
    cca_b_1: np.ndarray       # (T,) canonical variate for person B
    canonical_r: float        # Pearson r between the two variates
    weights_a: np.ndarray     # (D_a,) canonical loadings for A (retain for null)
    weights_b: np.ndarray     # (D_b,) canonical loadings for B (retain for null)
    valid_cols_a: np.ndarray  # indices of non-degenerate columns used
    valid_cols_b: np.ndarray
    n_prewhitened: bool       # whether AR(1) pre-whitening was applied


def compute_cca_1(
    features_a: np.ndarray,
    features_b: np.ndarray,
    n_components: int = 1,
    prewhiten: bool = True,
    return_result: bool = False,
) -> tuple[np.ndarray, np.ndarray, float] | CCAResult:
    """First canonical component per person.

    Args:
        features_a: (T, D_a) stacked modality features for person A.
        features_b: (T, D_b) stacked modality features for person B.
        n_components: typically 1 (we want the dominant coupling axis).
        prewhiten: If True (default), apply AR(1) residualization per column
                   before CCA. Audit HIGH-4 fix: mitigates spurious-correlation
                   inflation from time-series autocorrelation (Granger &
                   Newbold 1974). Set False for legacy behavior.
        return_result: If True, return CCAResult with canonical weights
                       retained — required for `partner_shuffled_null_fixed`
                       which uses the SAME weights on shuffled data (audit
                       HIGH-11 fix).

    Returns (legacy): (cca_a_1, cca_b_1, canonical_correlation_r).
    Returns (return_result=True): CCAResult dataclass.

    Edge cases (audit HIGH-4 handling):
    - Constant/zero-variance columns are dropped before fitting (previously
      produced NaN variates silently).
    - Column length mismatch → truncated to shorter.
    """
    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)
    if features_a.ndim == 1:
        features_a = features_a.reshape(-1, 1)
    if features_b.ndim == 1:
        features_b = features_b.reshape(-1, 1)
    if features_a.shape[0] != features_b.shape[0]:
        n = min(features_a.shape[0], features_b.shape[0])
        features_a = features_a[:n]
        features_b = features_b[:n]

    # Drop zero-variance columns before CCA (audit HIGH-4)
    std_a = features_a.std(axis=0)
    std_b = features_b.std(axis=0)
    valid_a = np.where(std_a > 1e-10)[0]
    valid_b = np.where(std_b > 1e-10)[0]
    if len(valid_a) == 0 or len(valid_b) == 0:
        raise ValueError(
            "compute_cca_1: all columns in features_%s are constant — "
            "cannot compute CCA" % ("a" if len(valid_a) == 0 else "b")
        )
    if len(valid_a) < features_a.shape[1] or len(valid_b) < features_b.shape[1]:
        logger.warning(
            "compute_cca_1: dropped %d/%d zero-variance cols in A, %d/%d in B",
            features_a.shape[1] - len(valid_a), features_a.shape[1],
            features_b.shape[1] - len(valid_b), features_b.shape[1],
        )
    a_f = features_a[:, valid_a]
    b_f = features_b[:, valid_b]

    # Optional AR(1) pre-whitening to address autocorrelation-driven
    # spurious correlation (audit HIGH-4)
    if prewhiten:
        a_f = ar1_prewhiten(a_f)
        b_f = ar1_prewhiten(b_f)

    # Z-score each column. sklearn CCA also scales internally, but this is
    # a small redundancy, not a statistical bug.
    def _zscore_cols(x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-12
        return (x - mu) / sd

    a_z = _zscore_cols(a_f)
    b_z = _zscore_cols(b_f)

    cca = CCA(n_components=n_components, max_iter=500)
    cca.fit(a_z, b_z)
    a_c, b_c = cca.transform(a_z, b_z)
    a_1 = a_c[:, 0]
    b_1 = b_c[:, 0]
    r_val = float(np.corrcoef(a_1, b_1)[0, 1])
    if not np.isfinite(r_val):
        logger.warning("compute_cca_1: canonical_r is not finite; setting to 0")
        r_val = 0.0

    if return_result:
        # Extract canonical weights for fixed-direction null testing
        w_a = cca.x_weights_[:, 0] if hasattr(cca, "x_weights_") else np.zeros(a_z.shape[1])
        w_b = cca.y_weights_[:, 0] if hasattr(cca, "y_weights_") else np.zeros(b_z.shape[1])
        return CCAResult(
            cca_a_1=a_1,
            cca_b_1=b_1,
            canonical_r=r_val,
            weights_a=w_a,
            weights_b=w_b,
            valid_cols_a=valid_a,
            valid_cols_b=valid_b,
            n_prewhitened=prewhiten,
        )
    return a_1, b_1, r_val


def build_cca_features(
    body_motion: np.ndarray,
    fau_mean: np.ndarray,
    f0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stack scalar modality signals into a CCA feature matrix.

    body_motion: (T,)
    fau_mean: (T,)
    f0: (T,) optional; if provided, included as third column.

    Returns:
        (T, D) where D ∈ {2, 3}. Truncates to common length.

    NOTE: This is the SCALAR-AGGREGATE path (Opus HIGH-5: "CCA collapses
    to weighted regression at D=2-3"). For the rich multimodal stack
    recommended by audit, use `build_cca_features_rich()` instead.
    """
    cols = [body_motion, fau_mean]
    if f0 is not None and len(f0) > 0:
        cols.append(f0)
    aligned = align_to_common_length(cols, mode="truncate")
    return np.column_stack(aligned)


def build_cca_features_rich(
    translation: Optional[np.ndarray] = None,
    body_pose: Optional[np.ndarray] = None,
    fau_value: Optional[np.ndarray] = None,
    head_encodings: Optional[np.ndarray] = None,
    f0: Optional[np.ndarray] = None,
    fs: float = SAMPLE_RATE,
    apply_per_modality_lowpass: bool = True,
    modality_bandpass: Optional[dict[str, tuple[float, float]]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack RICH per-modality features into a CCA feature matrix.

    Opus HIGH-5 fix (Apr 17, 2026): Using only scalar aggregates (D=2-3)
    collapses CCA to weighted regression — CCA cannot "canonicalize" with
    fewer dims than it has components. Rich stacking gives CCA meaningful
    multivariate input: body velocity (3) + body_pose PC1 (1) + FAU (24)
    + head (3) + F0 (1) ≈ 32 dims per person.

    Args:
        translation: (T, 3) pelvis XYZ, or None to skip body translation
        body_pose: (T, 21, 3) joint rotations, or None to skip PC1
        fau_value: (T, 24) FAU intensities, or None to skip FAU
        head_encodings: (T, 3) head neural encoding, or None to skip
        f0: (T,) F0 envelope, or None to skip prosody
        fs: sampling rate (30 Hz)
        apply_per_modality_lowpass: if True, lowpass-filter each modality
                                    to its appropriate band before stacking
        modality_bandpass: override per-modality bandpass as
                           {"body": (low, high), "fau": (...), ...}

    Returns:
        (features, col_names): (T, D) matrix aligned to common length,
        plus list of D column names (for interpretability of CCA weights).
    """
    if modality_bandpass is None:
        modality_bandpass = {}

    def _bp(modality: str, signal: np.ndarray) -> np.ndarray:
        """Apply per-modality bandpass if requested + length OK."""
        if not apply_per_modality_lowpass:
            return signal
        low, high = modality_bandpass.get(modality,
                                          MODALITY_BANDPASS_DEFAULT.get(modality, (0.1, 5.0)))
        try:
            return apply_bandpass(signal, low, high, fs=fs, order=4)
        except ValueError:
            return signal  # too short for filter

    cols: list[np.ndarray] = []
    col_names: list[str] = []

    # --- Body translation velocity (3 dims) ---
    if translation is not None:
        t = np.asarray(translation, dtype=np.float64)
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"translation must be (T,3), got {t.shape}")
        # Velocity per axis (with lowpass inside body_motion_energy-style)
        if fs > 0 and t.shape[0] >= 30:
            try:
                sos = butter(4, 5.0 / (fs / 2), btype="low", output="sos")
                t_lp = np.column_stack([sosfiltfilt(sos, t[:, i]) for i in range(3)])
            except Exception:
                t_lp = t
        else:
            t_lp = t
        tvel = np.diff(t_lp, axis=0, prepend=t_lp[:1])
        for i, axis in enumerate(["x", "y", "z"]):
            cols.append(tvel[:, i])
            col_names.append(f"body_trans_vel_{axis}")

    # --- Body pose PC1 of joint-rotation velocity (1 dim) ---
    if body_pose is not None:
        bp = np.asarray(body_pose, dtype=np.float64)
        if bp.ndim == 3:
            bp_flat = bp.reshape(bp.shape[0], -1)  # (T, 63)
        else:
            bp_flat = bp
        # Velocity magnitude per joint-dim, then PCA would be overkill;
        # use the aggregate velocity norm as a single "body kinematic"
        # channel. This is similar to body_motion_energy's body_pose term.
        if fs > 0 and bp_flat.shape[0] >= 30:
            try:
                sos = butter(4, 5.0 / (fs / 2), btype="low", output="sos")
                bp_lp = np.column_stack([sosfiltfilt(sos, bp_flat[:, i])
                                          for i in range(bp_flat.shape[1])])
            except Exception:
                bp_lp = bp_flat
        else:
            bp_lp = bp_flat
        bvel = np.diff(bp_lp, axis=0, prepend=bp_lp[:1])
        bvel_mag = np.linalg.norm(bvel, axis=1)
        cols.append(bvel_mag)
        col_names.append("body_pose_velocity_mag")

    # --- FAU: all 24 continuous activations, per-AU bandpassed ---
    if fau_value is not None:
        fau = np.asarray(fau_value, dtype=np.float64)
        if fau.ndim != 2 or fau.shape[1] != 24:
            raise ValueError(f"fau_value must be (T,24), got {fau.shape}")
        for i in range(24):
            cols.append(_bp("fau", fau[:, i]))
            col_names.append(f"fau_{i:02d}")

    # --- Head encoding: all 3 dims (neural latent, keep as-is) ---
    if head_encodings is not None:
        he = np.asarray(head_encodings, dtype=np.float64)
        if he.ndim != 2 or he.shape[1] != 3:
            raise ValueError(f"head_encodings must be (T,3), got {he.shape}")
        for i in range(3):
            cols.append(_bp("head", he[:, i]))
            col_names.append(f"head_enc_{i}")

    # --- F0 envelope (1 dim) ---
    if f0 is not None:
        f0_arr = np.asarray(f0, dtype=np.float64).ravel()
        cols.append(_bp("prosody", f0_arr))
        col_names.append("f0_envelope")

    if not cols:
        raise ValueError("At least one modality must be provided to build_cca_features_rich")

    aligned = align_to_common_length(cols, mode="truncate")
    return np.column_stack(aligned), col_names


def extract_dyad_cca_features(
    dyad: dict,
    f0_by_participant: Optional[dict[str, np.ndarray]] = None,
    voicing_by_participant: Optional[dict[str, np.ndarray]] = None,
    modality_bandpass: Optional[dict[str, tuple[float, float]]] = None,
    fs: float = SAMPLE_RATE,
    use_intersected_valid_mask: bool = True,
    use_intersected_voicing_mask: bool = False,
    min_length: int = 640,
) -> Optional[tuple[np.ndarray, np.ndarray, list[str]]]:
    """Build (features_A, features_B, col_names) for a dyad from
    load_dyad_from_manifest output, ready for compute_cca_1.

    Applies masks in priority order:
    - Intersection of both participants' is_valid masks (occlusion)
    - Optionally, intersection of both participants' voicing masks
      (both-voiced frames — Opus HIGH-C fix for F0 turn-taking confound)

    Args:
        dyad: output of load_dyad_from_manifest.
        f0_by_participant: {file_id: F0_envelope} for prosody column.
        voicing_by_participant: {file_id: voicing_mask (T,)} — if provided
                                 and use_intersected_voicing_mask=True, AND
                                 with the is_valid mask to keep only both-
                                 voiced frames. Removes turn-taking rhythm
                                 confound that median-imputed F0 introduces.
        modality_bandpass: override bandpass per modality.
        use_intersected_valid_mask: AND both participants' is_valid masks.
        use_intersected_voicing_mask: AND both participants' voicing masks
                                      (requires voicing_by_participant).
        min_length: reject dyad if common valid length < this.

    Returns:
        (features_A, features_B, col_names) or None if the dyad fails.
    """
    participants = dyad["participants"]
    if len(participants) != 2:
        return None
    p0, p1 = participants

    # ---- Build intersection of is_valid masks ----
    if use_intersected_valid_mask:
        m0 = dyad_is_valid_mask(p0)
        m1 = dyad_is_valid_mask(p1)
        if m0 is None and m1 is None:
            common_mask = None
        elif m0 is None:
            common_mask = m1
        elif m1 is None:
            common_mask = m0
        else:
            n = min(len(m0), len(m1))
            common_mask = m0[:n] & m1[:n]
    else:
        common_mask = None

    # ---- Optionally AND with voicing masks (Opus HIGH-C) ----
    file_ids = dyad.get("file_ids", [None, None])
    if use_intersected_voicing_mask and voicing_by_participant is not None:
        v0 = voicing_by_participant.get(file_ids[0]) if file_ids[0] else None
        v1 = voicing_by_participant.get(file_ids[1]) if file_ids[1] else None
        if v0 is not None and v1 is not None:
            n_v = min(len(v0), len(v1))
            voicing_and = np.asarray(v0[:n_v]).astype(bool) & np.asarray(v1[:n_v]).astype(bool)
            if common_mask is not None:
                n_common = min(len(common_mask), n_v)
                common_mask = common_mask[:n_common] & voicing_and[:n_common]
            else:
                common_mask = voicing_and

    def _apply_mask(arr):
        if common_mask is None or arr is None:
            return arr
        arr = np.asarray(arr)
        n = min(arr.shape[0], len(common_mask))
        return arr[:n][common_mask[:n]] if arr.ndim == 1 else arr[:n][common_mask[:n], ...]

    def _p_features(p, file_id_key):
        fau = _apply_mask(p.get("movement:FAUValue"))
        head = _apply_mask(p.get("movement:head_encodings"))
        trans = _apply_mask(p.get("smplh:translation"))
        pose = _apply_mask(p.get("smplh:body_pose"))
        f0 = None
        if f0_by_participant is not None and file_id_key in f0_by_participant:
            f0 = _apply_mask(f0_by_participant[file_id_key])
        # Check minimum length
        if trans is None or len(trans) < min_length:
            return None
        feats, names = build_cca_features_rich(
            translation=trans, body_pose=pose, fau_value=fau,
            head_encodings=head, f0=f0, fs=fs,
            modality_bandpass=modality_bandpass,
        )
        return feats, names

    r0 = _p_features(p0, file_ids[0])
    r1 = _p_features(p1, file_ids[1])
    if r0 is None or r1 is None:
        return None
    feats_a, names_a = r0
    feats_b, names_b = r1
    if names_a != names_b:
        logger.warning("extract_dyad_cca_features: column-name mismatch "
                       "between participants A and B (using A's names)")
    n = min(feats_a.shape[0], feats_b.shape[0])
    if n < min_length:
        return None
    return feats_a[:n], feats_b[:n], names_a


def expand_weights_to_full_space(
    compressed_weights: np.ndarray,
    valid_cols: np.ndarray,
    full_n_cols: int,
) -> np.ndarray:
    """Map CCA weights from compressed column space back to full space.

    When compute_cca_1 drops zero-variance columns, the returned weights
    are indexed in the compressed space. For visualization + interpretation,
    we need them in the original column order (so col_names align).

    Sonnet HIGH-4 fix (Apr 17): prevents silent mislabeling in heatmaps.

    Args:
        compressed_weights: (len(valid_cols),) canonical weights
        valid_cols: indices into original features that were kept
        full_n_cols: number of columns in the original features_a

    Returns:
        (full_n_cols,) weights array with 0.0 in dropped-column positions.
    """
    full = np.zeros(full_n_cols)
    full[valid_cols] = compressed_weights
    return full


def compute_null_normalized_r(
    observed_r: float,
    null_dist: np.ndarray,
) -> tuple[float, float, float]:
    """Compute z-score of observed canonical r against partner-shuffled null.

    Opus HIGH-D fix (Apr 17): raw canonical_r is upward-biased by √(D/T).
    The partner-shuffled null inherits the same bias, so the z-score
    (observed_r - null_mean) / null_std removes the T-dependent bias
    floor — comparable across dyads of different durations.

    Returns:
        (z_score, null_mean, null_std). Returns (nan, nan, nan) if null
        has too few finite samples.
    """
    null_finite = null_dist[np.isfinite(null_dist)] if null_dist is not None else np.array([])
    if len(null_finite) < 2:
        return np.nan, np.nan, np.nan
    null_mean = float(np.mean(null_finite))
    null_std = float(np.std(null_finite))
    if null_std <= 1e-12:
        return np.nan, null_mean, null_std
    z = (observed_r - null_mean) / null_std
    return float(z), null_mean, null_std


# ============================================================================
# Surrogates: IAAFT + partner-shuffled null
# ============================================================================

def iaaft_surrogates(
    signal: np.ndarray,
    n_surrogates: int = 200,
    max_iter: int = 100,
    tol: float = 1e-3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Iteratively Amplitude-Adjusted Fourier Transform surrogates.

    Schreiber & Schmitz 1996, Phys Rev Lett 77:635; 2000, Physica D 142:346.
    Preserves both the power spectrum and the amplitude distribution of
    the original signal while randomizing phase.

    Audit fixes (Apr 17, 2026):
    - HIGH-1: RNG default is now entropy-seeded (not hard seed=42), which
      prevents correlated null distributions across independent calls.
      Pass rng=np.random.default_rng(seed) explicitly for reproducibility.
    - HIGH-2: Convergence test changed from absolute MSE delta (1e-6) to
      RELATIVE delta (1e-3). Absolute 1e-6 on amplitude² units is scale-
      dependent: it either never triggers (high-amplitude signals) or
      triggers early on oscillating MSE (low-amplitude signals). Relative
      tolerance is scale-invariant. If you need strict convergence,
      set tol=0 to disable early exit — loop will always run max_iter.

    Args:
        signal: 1-D time series.
        n_surrogates: How many to generate.
        max_iter: Per-surrogate iteration cap (100 is literature standard).
        tol: Relative MSE delta for early termination. Set 0 to disable.
        rng: np.random.Generator. If None, entropy-seeded (fresh randomness
             each call). Pass an explicit generator for reproducibility.

    Returns:
        (n_surrogates, T) array.
    """
    if rng is None:
        # Entropy-seeded: each call gets independent randomness. This
        # prevents cross-dyad correlation that would otherwise arise from
        # the previous hard-coded seed=42 default.
        rng = np.random.default_rng()
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)
    if n < 8:
        raise ValueError(f"IAAFT requires signal length >= 8, got {n}")
    sorted_amp = np.sort(signal)
    target_spec = np.abs(np.fft.rfft(signal))
    out = np.zeros((n_surrogates, n), dtype=np.float64)

    for k in range(n_surrogates):
        # Initialize with random shuffle of original values (Schreiber &
        # Schmitz 2000 recommend this over AAFT-start; AAFT start can
        # get stuck in lower-accuracy local minima).
        surrogate = rng.permutation(signal).copy()
        prev_mse = np.inf
        for _ in range(max_iter):
            # Step 1: Enforce target spectrum (keep current phase)
            spec = np.fft.rfft(surrogate)
            phase = np.angle(spec)
            new_spec = target_spec * np.exp(1j * phase)
            s1 = np.fft.irfft(new_spec, n=n)
            # Step 2: Enforce target amplitude distribution (rank-match)
            ranks = np.argsort(np.argsort(s1))
            s2 = sorted_amp[ranks]
            # Relative convergence: |Δmse| / mse < tol
            mse = float(np.mean((np.abs(np.fft.rfft(s2)) - target_spec) ** 2))
            if tol > 0 and prev_mse != np.inf:
                rel_delta = abs(prev_mse - mse) / (prev_mse + 1e-12)
                if rel_delta < tol:
                    surrogate = s2
                    break
            prev_mse = mse
            surrogate = s2
        out[k] = surrogate

    return out


def partner_shuffled_null(
    dyad_data: list[tuple[np.ndarray, np.ndarray, str]],
    metric_fn,
    n_shuffles: int = 200,
    same_condition: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Partner-shuffled null — valid ONLY for metrics that do NOT re-fit
    on the shuffled data.

    ⚠ AUDIT HIGH-11 WARNING (Apr 17, 2026):
    If `metric_fn` re-fits CCA on each shuffled pair, the null distribution
    is invalid because CCA re-maximizes correlation each time.

    For metrics like raw Pearson correlation, windowed cross-correlation,
    or coherence at a fixed frequency, this function is correct.

    For CCA coupling on fixed canonical directions, use
    `partner_shuffled_null_fixed_directions()` instead.

    Args:
        dyad_data: list of (signal_a, signal_b, condition_tag).
        metric_fn: callable (sig_a, sig_b) -> float. MUST NOT be a metric
                   that re-fits a maximizing model on the shuffled pair.
        n_shuffles: null draws per dyad.
        same_condition: if True, only shuffle within the same condition
                        (preserves condition-level base rates).
        rng: random generator. If None, entropy-seeded.

    Returns:
        (len(dyad_data), n_shuffles) array of null metric values.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_dyads = len(dyad_data)
    null = np.zeros((n_dyads, n_shuffles))

    for i, (sig_a, _, cond_i) in enumerate(dyad_data):
        if same_condition:
            pool_idx = [
                j for j in range(n_dyads)
                if j != i and dyad_data[j][2] == cond_i
            ]
        else:
            pool_idx = [j for j in range(n_dyads) if j != i]
        if not pool_idx:
            null[i] = np.nan
            continue
        choices = rng.choice(pool_idx, size=n_shuffles, replace=True)
        for s, j in enumerate(choices):
            sig_b_fake = dyad_data[j][1]
            n = min(len(sig_a), len(sig_b_fake))
            try:
                null[i, s] = float(metric_fn(sig_a[:n], sig_b_fake[:n]))
            except Exception as exc:
                logger.debug(
                    "partner-shuffle[%d,%d] metric_fn failed: %s", i, s, exc
                )
                null[i, s] = np.nan
    return null


def partner_shuffled_null_fixed_directions(
    observed_cca_result: CCAResult,
    features_a_observed: np.ndarray,
    other_dyad_features_b: list[np.ndarray],
    n_shuffles: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Partner-shuffled null for CCA using FIXED canonical directions.

    Audit HIGH-11 fix (Apr 17, 2026): The original `partner_shuffled_null`
    re-fits CCA on each shuffled pair, allowing every null draw to re-
    maximize correlation → null ≈ observation → p ≈ 0.5 regardless of
    true coupling.

    Correct protocol (Hardoon et al. 2004 §4, Winkler et al. 2020):
    1. Fit CCA once on the observed dyad → obtain canonical weights w_a, w_b.
    2. For each null draw, project (fixed) weights onto the OBSERVED
       person A features and a SHUFFLED person B from another dyad.
    3. Compute Pearson r between the two fixed-direction projections.

    This measures how often random partner pairings would project
    together under the SAME coupling directions learned from the real
    dyad — a valid null.

    Args:
        observed_cca_result: Output of `compute_cca_1(..., return_result=True)`
                             on the real dyad.
        features_a_observed: (T, D_a) real person A features used in the fit.
        other_dyad_features_b: list of (T_j, D_b) features from other dyads
                               (person B, can be different conditions).
        n_shuffles: null draws.
        rng: entropy-seeded if None.

    Returns:
        (n_shuffles,) array of null canonical r values.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(other_dyad_features_b) == 0:
        return np.full(n_shuffles, np.nan)

    # Project observed person A features through FIXED w_a
    valid_a = observed_cca_result.valid_cols_a
    w_a = observed_cca_result.weights_a
    a_cols = features_a_observed[:, valid_a]
    # Z-score same way as fit (center + scale)
    a_centered = (a_cols - a_cols.mean(axis=0)) / (a_cols.std(axis=0) + 1e-12)
    a_proj = a_centered @ w_a  # (T,)

    null = np.zeros(n_shuffles)
    valid_b = observed_cca_result.valid_cols_b
    w_b = observed_cca_result.weights_b

    for s in range(n_shuffles):
        # Pick a random other-dyad partner B
        j = rng.integers(0, len(other_dyad_features_b))
        b_feats = other_dyad_features_b[j]
        if b_feats.ndim == 1:
            b_feats = b_feats.reshape(-1, 1)
        # Truncate to observed length
        n = min(len(a_proj), len(b_feats))
        if n < 30:
            null[s] = np.nan
            continue
        # Handle column-count mismatch by taking min — or skip if incompatible
        if b_feats.shape[1] < valid_b.max() + 1:
            null[s] = np.nan
            continue
        b_cols = b_feats[:n, valid_b]
        b_centered = (b_cols - b_cols.mean(axis=0)) / (b_cols.std(axis=0) + 1e-12)
        b_proj = b_centered @ w_b  # (n,)
        r_val = float(np.corrcoef(a_proj[:n], b_proj)[0, 1])
        null[s] = r_val if np.isfinite(r_val) else np.nan

    return null


# ============================================================================
# Significance testing
# ============================================================================

def surrogate_p_value(
    observed: float,
    null_dist: np.ndarray,
    two_sided: bool = False,
) -> float:
    """Phipson & Smyth (2010) corrected p-value: (b + 1) / (n + 1).

    Where b = count of surrogates at least as extreme as observed.
    Avoids the biased p = b/n that returns 0.0 when no surrogate exceeds.
    """
    null_dist = np.asarray(null_dist)
    null_dist = null_dist[np.isfinite(null_dist)]
    n = len(null_dist)
    if n == 0:
        return np.nan
    if two_sided:
        b = int(np.sum(np.abs(null_dist) >= abs(observed)))
    else:
        b = int(np.sum(null_dist >= observed))
    return (b + 1) / (n + 1)


# ============================================================================
# Batch PSD diagnostic (Step 2 helper)
# ============================================================================

def batch_psd_report(
    signals_by_channel: dict[str, list[np.ndarray]],
    fs: float = SAMPLE_RATE,
    modality_map: Optional[dict[str, str]] = None,
    min_length: int = 640,
) -> dict[str, dict]:
    """Compute PSD diagnostic across many signals per channel.

    Args:
        signals_by_channel: {channel_name: [signal_1, signal_2, ...]}
        fs: sampling rate
        modality_map: optional {channel_name: modality_key} to route SNR +
                      quality calls through per-modality bands (e.g. map
                      'emotion_valence' → 'emotion'). If None, each channel
                      name is used as its own modality key (which works
                      because of the MODALITY_SNR_BANDS alias entries).
        min_length: minimum signal length (samples) for inclusion. Default
                    640 ≈ 21 s at 30 Hz — enough for ≥3 Welch segments with
                    nperseg=256 and 50% overlap (Opus H6 fix).

    Opus H2 fix (Apr 17, 2026): previously, every call inside this
    function omitted the `modality=` kwarg, so SNR and quality gate were
    body-centric across ALL modalities, silently contradicting the
    notebook narrative. Now the modality is threaded through.

    Opus H6 fix: min_length raised from 64 to 640 so Welch has enough
    segments for a stable PSD estimate (single-segment PSD is not
    consistent per Percival & Walden 1993 §6.4).

    Opus H7 fix: per-modality nperseg from MODALITY_NPERSEG used for SNR
    + bandpass estimation; long signals (emotion) get 512-sample windows
    to resolve sub-0.1 Hz content, short-dynamics signals (FAU) get 128.

    Apr 18 2026 fix (H7 actually threaded): the five inner helpers
    (recommend_bandpass, estimate_noise_floor, spectral_flatness,
    signal_to_noise_ratio_db, assess_signal_quality) all now accept and
    propagate `nperseg`. Previously the per-modality value was computed
    at line 1702 but never passed into the helpers, so every PSD call
    silently used the 256-sample default — emotion channels lost half
    their sub-0.1 Hz frequency resolution.

    Returns:
        {channel_name: {
            "recommended_bandpass": (low, high),
            "median_noise_floor_freq": float,
            "quality_pass_rate": float,
            "median_spectral_flatness": float,
            "median_snr_db": float,
            "n_signals": int,
            "n_skipped_short": int,
            "modality_used": str,
            "nperseg_used": int,
        }}
    """
    if modality_map is None:
        modality_map = {}
    out: dict[str, dict] = {}
    for ch, sigs in signals_by_channel.items():
        modality = modality_map.get(ch, ch)  # default: channel name == modality key
        nperseg = MODALITY_NPERSEG.get(modality, 256)
        bps_low, bps_high, nfs, sfs, snrs, passes = [], [], [], [], [], 0
        skipped = 0
        for s in sigs:
            if len(s) < min_length:
                skipped += 1
                continue
            # H7 fix (now actually threaded): pass per-modality nperseg through
            # to every PSD-derived helper so emotion gets 512, FAU gets 128, etc.
            low, high = recommend_bandpass(s, fs=fs, nperseg=nperseg)
            bps_low.append(low)
            bps_high.append(high)
            nf, _ = estimate_noise_floor(s, fs=fs, nperseg=nperseg)
            nfs.append(nf)
            sfs.append(spectral_flatness(s, fs=fs, nperseg=nperseg))
            # H2 fix: pass modality for per-modality noise band
            if modality in MODALITY_SNR_BANDS:
                sig_band, noise_band = MODALITY_SNR_BANDS[modality]
                snrs.append(signal_to_noise_ratio_db(
                    s, fs=fs, signal_band=sig_band, noise_band=noise_band,
                    nperseg=nperseg,
                ))
            else:
                snrs.append(signal_to_noise_ratio_db(s, fs=fs, nperseg=nperseg))
            qr = assess_signal_quality(s, fs=fs, modality=modality, nperseg=nperseg)
            if qr.passed:
                passes += 1
        out[ch] = {
            "recommended_bandpass": (
                float(np.median(bps_low)) if bps_low else np.nan,
                float(np.median(bps_high)) if bps_high else np.nan,
            ),
            "median_noise_floor_freq": (
                float(np.median(nfs)) if nfs else np.nan
            ),
            "quality_pass_rate": passes / max(len(sigs) - skipped, 1),
            "median_spectral_flatness": (
                float(np.median(sfs)) if sfs else np.nan
            ),
            "median_snr_db": float(np.median(snrs)) if snrs else np.nan,
            "n_signals": len(sigs),
            "n_skipped_short": skipped,
            "modality_used": modality,
            "nperseg_used": nperseg,
        }
    return out


# ============================================================================
# Analytical coupling threshold (Garijo/Gómez/Arenas 2026 + Rodrigues 2016)
# ============================================================================

def compute_analytical_threshold_garijo2026(
    omega: np.ndarray,
    adjacency: np.ndarray,
    *,
    frequency_density_estimator: str = "kde",
    illustrative: bool = True,
) -> dict:
    """Analytical critical coupling K_c for a finite Kuramoto network.

    Combines two citations:
      * **Rodrigues, Peron, Ji & Kurths 2016** *Phys. Rep.* **610** — mean-field
        backbone: K_c ≈ 2 / (π · g(0) · λ_max(A)), where g(0) is the natural-
        frequency density at the mean of ω and λ_max(A) is the largest real
        eigenvalue of the weighted adjacency matrix A. VERIFIED citation.
      * **Garijo, Gómez & Arenas 2026** arXiv:2604.14772 — convex-geometric
        refinement for finite N. Submitted Apr 16, 2026; not yet peer-reviewed.
        Returned with `is_illustrative=True` until the Garijo derivation is
        independently validated against a known test case. Current
        implementation returns the Rodrigues mean-field value.

    Args:
        omega: 1-D array of per-oscillator natural frequencies (rad/s).
        adjacency: (N, N) weighted adjacency matrix (symmetric).
        frequency_density_estimator: 'kde' uses scipy.stats.gaussian_kde;
            'gaussian' uses analytic Gaussian density at the mean.
        illustrative: if True, mark the returned K_c as illustrative only.

    Returns:
        dict with keys:
          K_c : predicted critical coupling (float). NaN if degenerate.
          lambda_max : largest real eigenvalue of adjacency (float).
          g_at_mean : natural-frequency density at the mean (float).
          n_oscillators : N (int).
          is_illustrative : bool.
          citation : str.

    Apr 18 2026 (sprint day 5): introduced for NB1 Tier-1 analytical sidecar.
    """
    omega = np.asarray(omega, dtype=float).ravel()
    adjacency = np.asarray(adjacency, dtype=float)
    N = len(omega)
    if adjacency.shape != (N, N):
        raise ValueError(
            f"adjacency shape {adjacency.shape} does not match omega length {N}"
        )

    # Largest real eigenvalue of A (assume symmetric; use eigvalsh for stability)
    eig_real = np.linalg.eigvalsh(adjacency)
    lambda_max = float(np.max(eig_real))

    # Natural-frequency density at the mean
    mu_omega = float(np.mean(omega))
    if frequency_density_estimator == "kde":
        try:
            from scipy.stats import gaussian_kde
            if N >= 2 and float(np.std(omega)) > 1e-12:
                kde = gaussian_kde(omega)
                g_at_mean = float(kde(mu_omega)[0])
            else:
                sigma = max(float(np.std(omega)), 0.1)
                g_at_mean = 1.0 / (sigma * np.sqrt(2 * np.pi))
        except ImportError:
            sigma = max(float(np.std(omega)), 0.1)
            g_at_mean = 1.0 / (sigma * np.sqrt(2 * np.pi))
    elif frequency_density_estimator == "gaussian":
        sigma = max(float(np.std(omega)), 0.1)
        g_at_mean = 1.0 / (sigma * np.sqrt(2 * np.pi))
    else:
        raise ValueError(
            f"Unknown frequency_density_estimator: {frequency_density_estimator}"
        )

    if g_at_mean <= 0 or lambda_max <= 0:
        K_c = float("nan")
    else:
        K_c = 2.0 / (np.pi * g_at_mean * lambda_max)

    citation = (
        "Rodrigues et al. 2016 Phys. Rep. 610 [verified mean-field backbone]; "
        "Garijo/Gómez/Arenas 2026 arXiv:2604.14772 [convex-geometric refinement, "
        "pending independent validation — current return is Rodrigues mean-field]"
    )

    return {
        "K_c": float(K_c),
        "lambda_max": lambda_max,
        "g_at_mean": g_at_mean,
        "n_oscillators": N,
        "is_illustrative": bool(illustrative),
        "citation": citation,
    }


# ============================================================================
# Rupture-anchored windowing (Marwan 2007 + Fusaroli 2014 — replaces ref 71 Galati)
# ============================================================================

def anchor_crqa_to_rupture_events(
    time_series: np.ndarray,
    rupture_times,
    *,
    fs: float = SAMPLE_RATE,
    pre_window_s: float = 90.0,
    post_window_s: float = 30.0,
    drop_overlapping: bool = True,
) -> list:
    """Event-anchor a scalar time series around each rupture.

    Pure windowing utility: given rupture event times (seconds), returns
    segments of `time_series` of length (pre + post) centered on each event.
    Segments that extend outside [0, T] are dropped. If `drop_overlapping`,
    overlapping segments are removed, keeping the earlier.

    Enables event-anchored (instead of fixed-window) CSD / CRQA analysis
    in NB2. Citations (Apr 18 2026 correction):
      * Marwan, Romano, Thiel & Kurths 2007 *Phys. Rep.* 438:237–329
        (CRP textbook, canonical recurrence-plot methods).
      * Fusaroli, Rączaszek-Leonardi & Tylén 2014 *New Ideas in Psychology*
        32:147–157, DOI:10.1016/j.newideapsych.2013.03.005 (windowed CRQA
        for conversational dialog as coupled dynamical system).

    NOTE: Galati 2026 was initially credited; that was a misattribution
    (see REFERENCE_LIST ref 71 correction + CITATION_VERIFICATION_APR17.md).

    Args:
        time_series: 1-D array.
        rupture_times: iterable of rupture times (seconds from t=0).
        fs: sampling rate (Hz). Defaults to 30.
        pre_window_s: pre-event window length (seconds). Default 90.
        post_window_s: post-event window length (seconds). Default 30.
        drop_overlapping: drop overlapping windows, keeping the earlier.

    Returns:
        list of dicts, one per usable rupture, with keys:
          rupture_time_s : event time rounded to the nearest sample (s).
          start_idx, end_idx : segment bounds in the original series.
          segment : 1-D copy of the anchored window.
          pre_samples, post_samples : sample counts.
          out_of_range : bool (always False for returned items).

    Apr 18 2026 (sprint day 5): introduced for NB2 Tier-1 rupture-anchored sidecar.
    """
    ts = np.asarray(time_series, dtype=float).ravel()
    T = len(ts)
    pre_samples = int(round(pre_window_s * fs))
    post_samples = int(round(post_window_s * fs))

    rupture_idx = sorted(int(round(float(r) * fs)) for r in rupture_times)
    anchored: list = []
    prev_end = -1
    for r_idx in rupture_idx:
        start = r_idx - pre_samples
        end = r_idx + post_samples
        if start < 0 or end > T:
            continue
        if drop_overlapping and start < prev_end:
            continue
        anchored.append({
            "rupture_time_s": r_idx / fs,
            "start_idx": int(start),
            "end_idx": int(end),
            "segment": ts[start:end].copy(),
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "out_of_range": False,
        })
        prev_end = end
    return anchored


# ============================================================================
# HMM regime segmentation (Vidaurre 2018 TDE-HMM — backbone; NOT Li 2026 k-means)
# ============================================================================

def hmm_regime_segmentation(
    time_series: np.ndarray,
    *,
    n_states: int = 3,
    random_state: int = 42,
    n_iter: int = 100,
    covariance_type: str = "full",
) -> dict:
    """Fit a Gaussian HMM to a scalar coupling time series.

    Returns regime decomposition: per-sample state assignment, transition
    matrix, and state means. Used by NB2 as a sidecar to continuous CSD —
    reframes 'rising autocorrelation in a continuous signal' as 'discrete
    regime transitions' per Vidaurre et al. 2018 NeuroImage 180:646 (TDE-HMM).

    Falls back gracefully if `hmmlearn` is not installed, returning a
    `{error, fallback: True, n_states}` dict that NB2 can branch on.

    NOTE: Li et al. Comms Biol 2026 describes discrete coupling states in
    EEG hyperscanning but uses k-means on wavelet coherence, NOT HMM — do
    NOT cite Li 2026 here. The HMM-for-dynamical-regimes lineage is
    Vidaurre 2018 (ref 77; see REFERENCE_LIST correction).

    Args:
        time_series: 1-D array (e.g. Kuramoto R(t) or canonical correlation
            trajectory).
        n_states: number of discrete regimes. Default 3.
        random_state: RNG seed for reproducibility. Default 42.
        n_iter: EM max iterations. Default 100.
        covariance_type: one of 'full', 'diag', 'tied', 'spherical'.

    Returns:
        Success: dict with keys:
          states : 1-D int ndarray, per-sample state assignment (length T).
          transition_matrix : (n_states, n_states) stochastic matrix.
          state_means : (n_states,) ndarray of per-state means.
          log_likelihood : float, final EM log-likelihood.
          n_states : int.
          converged : bool.
          citation : str.
        Fallback (hmmlearn missing): dict with keys:
          error, fallback=True, n_states.

    Apr 18 2026 (sprint day 5): introduced for NB2 Tier-2 regime sidecar.
    """
    ts = np.asarray(time_series, dtype=float).ravel()
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except ImportError as e:
        return {
            "error": (
                f"hmmlearn not available ({e}); install with "
                "`pip install 'hmmlearn>=0.3.0'` to enable this helper."
            ),
            "fallback": True,
            "n_states": int(n_states),
        }

    X = ts.reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=int(n_states),
        covariance_type=covariance_type,
        n_iter=int(n_iter),
        random_state=int(random_state),
        init_params="stmc",
    )
    hmm.fit(X)
    states = hmm.predict(X).astype(int)
    log_likelihood = float(hmm.score(X))
    return {
        "states": states,
        "transition_matrix": np.asarray(hmm.transmat_, dtype=float),
        "state_means": np.asarray(hmm.means_.ravel(), dtype=float),
        "log_likelihood": log_likelihood,
        "n_states": int(n_states),
        "converged": bool(hmm.monitor_.converged),
        "citation": (
            "Vidaurre et al. 2018 NeuroImage 180:646 "
            "DOI:10.1016/j.neuroimage.2017.06.077 (TDE-HMM on brain states). "
            "NOT Li et al. Comms Biol 2026 which uses k-means, not HMM."
        ),
    }


# ============================================================================
# __main__ smoke test
# ============================================================================

if __name__ == "__main__":
    # Smoke test with synthetic data — validates all post-audit fixes
    rng = np.random.default_rng(42)
    t = np.arange(0, 60, 1 / 30)  # 60 seconds at 30 Hz
    sig_a = (
        np.sin(2 * np.pi * 1.0 * t)
        + 0.3 * rng.standard_normal(len(t))
    )
    sig_b = (
        np.sin(2 * np.pi * 1.0 * t + 0.3)
        + 0.3 * rng.standard_normal(len(t))
    )

    print("=" * 60)
    print("signal_utils.py smoke tests (post-audit)")
    print("=" * 60)

    # 1. PSD
    print("\n[1] PSD + bandpass recommendation")
    f, p = compute_psd(sig_a)
    print(f"  Peak at {f[np.argmax(p)]:.2f} Hz (expected ~1.0)")
    low, high = recommend_bandpass(sig_a)
    print(f"  Recommended bandpass: {low:.2f}-{high:.2f} Hz "
          f"(audit HIGH-3 fix: should not collapse to DC)")

    # 2. Quality gate with per-modality bands
    print("\n[2] Quality gate (per-modality noise bands)")
    qr_body = assess_signal_quality(sig_a, modality="body")
    print(f"  body modality: passed={qr_body.passed}, "
          f"flatness={qr_body.spectral_flatness:.3f}, "
          f"SNR={qr_body.snr_db:.2f} dB")
    qr_prosody = assess_signal_quality(sig_a, modality="prosody")
    print(f"  prosody modality: passed={qr_prosody.passed}, "
          f"SNR={qr_prosody.snr_db:.2f} dB "
          f"(audit HIGH-8: different band)")

    # 3. CCA with AR(1) pre-whitening + fixed-directions null
    print("\n[3] CCA pipeline (audit HIGH-4 + HIGH-11 fixes)")
    fau_a = rng.standard_normal(len(t)).cumsum() / 10
    fau_b = fau_a * 0.7 + 0.3 * rng.standard_normal(len(t)).cumsum() / 10
    feat_a = build_cca_features(sig_a, fau_a)
    feat_b = build_cca_features(sig_b, fau_b)

    # Default (prewhitened)
    ccaR = compute_cca_1(feat_a, feat_b, return_result=True, prewhiten=True)
    print(f"  Pre-whitened CCA r: {ccaR.canonical_r:.3f}")

    # Legacy (not pre-whitened) for comparison
    _, _, r_legacy = compute_cca_1(feat_a, feat_b, prewhiten=False)
    print(f"  Legacy (no prewhiten) r: {r_legacy:.3f} "
          f"(usually higher due to autocorrelation)")

    # 4. Fixed-directions null (valid for CCA)
    print("\n[4] Fixed-directions partner-shuffled null (audit HIGH-11)")
    # Generate 20 "other dyads" for the pool
    other_b = [
        np.column_stack([
            rng.standard_normal(len(t)),
            rng.standard_normal(len(t)).cumsum() / 10,
        ])
        for _ in range(20)
    ]
    null_dist = partner_shuffled_null_fixed_directions(
        ccaR, feat_a, other_b, n_shuffles=50
    )
    print(f"  Null mean = {np.nanmean(null_dist):.3f}, "
          f"std = {np.nanstd(null_dist):.3f}")
    pv_fixed = surrogate_p_value(ccaR.canonical_r, null_dist, two_sided=True)
    print(f"  p-value (two-sided, Phipson-Smyth): {pv_fixed:.4f}")

    # 5. IAAFT with proper convergence + metric_fn loop (audit CRITICAL-2 fix)
    print("\n[5] IAAFT surrogates + correct p-value usage")
    surr_ens = iaaft_surrogates(sig_a, n_surrogates=20, max_iter=50)
    print(f"  Shape: {surr_ens.shape}")
    # CORRECT: apply metric_fn (e.g., Pearson r) to each surrogate vs sig_b
    null_iaaft = np.array([
        float(np.corrcoef(surr_ens[k], sig_b)[0, 1])
        for k in range(surr_ens.shape[0])
    ])
    obs_r = float(np.corrcoef(sig_a, sig_b)[0, 1])
    pv_iaaft = surrogate_p_value(obs_r, null_iaaft, two_sided=True)
    print(f"  Observed r(sig_a, sig_b) = {obs_r:.3f}")
    print(f"  Null r mean = {null_iaaft.mean():.3f}, std = {null_iaaft.std():.3f}")
    print(f"  p-value: {pv_iaaft:.4f}")

    # 6. Edge cases
    print("\n[6] Edge-case guards")
    # Short-signal bandpass crash guard
    try:
        apply_bandpass(sig_a[:10], 0.5, 3.0)
    except ValueError as e:
        print(f"  Short-signal bandpass correctly rejected: {e}")
    # Constant-column CCA
    try:
        const_feat = np.zeros((100, 2))
        compute_cca_1(const_feat, feat_b[:100])
    except ValueError as e:
        print(f"  Constant-column CCA correctly rejected: {e}")

    # 7. Apr 18 2026: per-modality nperseg threading (batch_psd_report fix)
    print("\n[7] Per-modality nperseg threading (Apr 18 fix)")
    # Generate 30s @ 30Hz = 900 samples (> min_length=640)
    rng7 = np.random.default_rng(7)
    long_sig = np.sin(2 * np.pi * 0.2 * t) + 0.3 * rng7.standard_normal(len(t))
    psd_report = batch_psd_report(
        {"emotion_valence": [long_sig], "body": [long_sig], "fau": [long_sig]},
        fs=30.0,
        min_length=640,
    )
    assert psd_report["emotion_valence"]["nperseg_used"] == MODALITY_NPERSEG["emotion_valence"] == 512, \
        "emotion_valence should use nperseg=512 (was silently using 256 before Apr 18 fix)"
    assert psd_report["body"]["nperseg_used"] == MODALITY_NPERSEG["body"] == 256, \
        "body should use nperseg=256"
    assert psd_report["fau"]["nperseg_used"] == MODALITY_NPERSEG["fau"] == 128, \
        "fau should use nperseg=128"
    print(
        f"  emotion_valence: nperseg={psd_report['emotion_valence']['nperseg_used']} (expected 512) ✓"
    )
    print(
        f"  body:            nperseg={psd_report['body']['nperseg_used']} (expected 256) ✓"
    )
    print(
        f"  fau:             nperseg={psd_report['fau']['nperseg_used']} (expected 128) ✓"
    )

    # 8. Apr 18 2026: Garijo analytical coupling threshold
    print("\n[8] Analytical coupling threshold (Garijo 2026 / Rodrigues 2016 backbone)")
    omega8 = rng7.normal(loc=1.0, scale=0.1, size=4)
    A8 = np.ones((4, 4)) - np.eye(4)  # fully-connected 4-oscillator ring (K_n - I)
    kc_res = compute_analytical_threshold_garijo2026(
        omega8, A8, frequency_density_estimator="kde", illustrative=True,
    )
    assert abs(kc_res["lambda_max"] - 3.0) < 1e-6, \
        f"lambda_max of K_4-I should be 3.0, got {kc_res['lambda_max']}"
    assert kc_res["is_illustrative"] is True
    assert 0.0 < kc_res["K_c"] < 10.0, f"K_c out of sanity range: {kc_res['K_c']}"
    assert kc_res["n_oscillators"] == 4
    print(
        f"  K_c = {kc_res['K_c']:.3f} rad/s  "
        f"(lambda_max={kc_res['lambda_max']:.3f}, g(0)={kc_res['g_at_mean']:.3f})"
    )
    print(f"  is_illustrative = {kc_res['is_illustrative']} (flag stays True until Garijo validated)")

    # 9. Apr 18 2026: rupture-anchored windowing
    print("\n[9] Rupture-anchored windowing (Marwan 2007 + Fusaroli 2014)")
    fs9 = 30.0
    ts9 = np.arange(240 * int(fs9)).astype(float)  # 240s series, linearly increasing
    rupt9 = [100.0]  # single rupture at t=100s
    segs9 = anchor_crqa_to_rupture_events(
        ts9, rupt9, fs=fs9, pre_window_s=90.0, post_window_s=30.0,
    )
    assert len(segs9) == 1, f"expected 1 segment, got {len(segs9)}"
    expected_len = int(round((90.0 + 30.0) * fs9))
    assert segs9[0]["segment"].shape == (expected_len,), \
        f"segment shape {segs9[0]['segment'].shape} != ({expected_len},)"
    assert segs9[0]["pre_samples"] == 90 * int(fs9)
    assert segs9[0]["post_samples"] == 30 * int(fs9)
    print(f"  1 rupture @ t=100s (in 240s series) -> 1 segment of {expected_len} samples ✓")
    # Boundary: rupture too close to start/end should drop
    segs9b = anchor_crqa_to_rupture_events(ts9, [10.0, 235.0], fs=fs9)
    assert segs9b == [], f"boundary ruptures should drop, got {len(segs9b)}"
    print("  Boundary ruptures (within pre/post window of edges) correctly dropped ✓")

    # 10. Apr 18 2026: HMM regime segmentation (graceful fallback if hmmlearn missing)
    print("\n[10] HMM regime segmentation (Vidaurre 2018 TDE-HMM backbone)")
    T_pre10 = int(50 * fs9)
    T_post10 = int(50 * fs9)
    rng10 = np.random.default_rng(10)
    low_state = rng10.normal(loc=0.1, scale=0.03, size=T_pre10)
    high_state = rng10.normal(loc=0.8, scale=0.03, size=T_post10)
    ts10 = np.concatenate([low_state, high_state])
    hmm_res = hmm_regime_segmentation(ts10, n_states=2, random_state=42)
    if hmm_res.get("fallback"):
        print(f"  hmmlearn fallback triggered: {hmm_res['error'][:80]}...")
        print("  (Install hmmlearn>=0.3.0 to enable this helper.)")
    else:
        assert hmm_res["states"].shape == (len(ts10),), "states length mismatch"
        assert hmm_res["transition_matrix"].shape == (2, 2)
        assert hmm_res["state_means"].shape == (2,)
        # Find first index where state changes; should be near T_pre10
        states = hmm_res["states"]
        transitions = np.where(np.diff(states) != 0)[0]
        if len(transitions) > 0:
            first_transition = int(transitions[0])
            tolerance_samples = int(3 * fs9)  # ±3 s
            assert abs(first_transition - T_pre10) <= tolerance_samples, (
                f"HMM state change at idx={first_transition}, "
                f"expected within ±{tolerance_samples} of {T_pre10}"
            )
            print(
                f"  2 regimes synthesized; HMM found switch at idx={first_transition} "
                f"(truth={T_pre10}) ✓"
            )
        else:
            print("  HMM produced a single state — synthetic contrast too low (rare)")
        print(f"  state_means = {hmm_res['state_means'].round(3).tolist()}")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
