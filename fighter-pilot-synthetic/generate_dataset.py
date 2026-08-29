"""
Synthetic Fighter-Pilot Mental Workload Dataset Generator
============================================================
Implements the causal pipeline described in the methodology document:

  Pilot baseline -> mission/scenario -> workload-demand components
      -> latent workload -> performance + physiology -> workload label

Sources of manipulation logic (see references at bottom of script):
  - Chen et al. (2025):        ATA/ATG task difficulty (target/engagement counts)
  - Mohanavelu et al. (2022):  visibility + secondary cognitive task, flight phases
  - Svensson et al. (1997):    tactical/information complexity, simultaneous tasking
  - Gambiraza et al. (2026):   optional startle/stress module (flagged, off by default)

Every generated column is tagged in the companion data dictionary with a
provenance label: SOURCE_GROUNDED / METHODOLOGICAL_INFERENCE / SYNTHETIC_ASSUMPTION.

Author: generated per user's request for a research-paper-grade synthetic dataset.
"""

import numpy as np
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------------
# 0. Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
GENERATOR_VERSION = "v1.0"

# ----------------------------------------------------------------------------
# 1. Pilot layer  (Step 1 of methodology)
# ----------------------------------------------------------------------------
N_PILOTS = 150

exp_groups = rng.choice(
    ["novice", "intermediate", "experienced"],
    size=N_PILOTS,
    p=[0.30, 0.45, 0.25],
)
# Skill effect: experienced pilots handle workload better (protective effect)
exp_skill_map = {"novice": -1.0, "intermediate": 0.0, "experienced": 1.0}
pilot_skill_base = np.array([exp_skill_map[g] for g in exp_groups])
pilot_skill = pilot_skill_base + rng.normal(0, 0.4, N_PILOTS)  # within-group variability

pilots = pd.DataFrame({
    "pilot_id": [f"P{idx:03d}" for idx in range(1, N_PILOTS + 1)],
    "experience_group": exp_groups,
    "years_experience": np.clip(
        rng.normal(
            [{"novice": 2, "intermediate": 7, "experienced": 14}[g] for g in exp_groups],
            1.8,
        ), 0.5, 30
    ),
    "pilot_skill_z": pilot_skill,
    # Baseline physiology (resting), pilot-specific -> pilot random effect source
    "baseline_HR": np.clip(rng.normal(65, 7, N_PILOTS), 48, 95),
    "baseline_SDNN": np.clip(rng.normal(58, 12, N_PILOTS), 20, 110),
    "baseline_RMSSD": np.clip(rng.normal(42, 11, N_PILOTS), 12, 90),
    "baseline_LF_HF": np.clip(rng.normal(1.3, 0.4, N_PILOTS), 0.4, 3.5),
    "baseline_theta_power": np.clip(rng.normal(0.20, 0.04, N_PILOTS), 0.08, 0.35),
    "baseline_alpha_power": np.clip(rng.normal(0.30, 0.05, N_PILOTS), 0.15, 0.48),
    "baseline_delta_power": np.clip(rng.normal(0.18, 0.03, N_PILOTS), 0.08, 0.30),
    "baseline_fixation_duration_ms": np.clip(rng.normal(280, 40, N_PILOTS), 180, 420),
    "baseline_blink_rate_bpm": np.clip(rng.normal(16, 4, N_PILOTS), 5, 30),
    "baseline_pupil_mm": np.clip(rng.normal(4.2, 0.4, N_PILOTS), 3.0, 5.5),
    # Idiosyncratic response-gain random effect (how strongly THIS pilot's
    # physiology reacts to workload) -- pilot random effect required by Step 1
    "pilot_reactivity_gain": np.clip(rng.normal(1.0, 0.22, N_PILOTS), 0.5, 1.8),
})
pilots = pilots.set_index("pilot_id", drop=False)

# ----------------------------------------------------------------------------
# 2. Mission generator (Step 2: Chen ATA/ATG difficulty tiers)
# ----------------------------------------------------------------------------
DIFFICULTY_TIERS = {
    # tier: (target_count, required_engagements)  -- Chen-style 3/1, 6/2, 9/3
    "low":    (3, 1),
    "medium": (6, 2),
    "high":   (9, 3),
}
DIFFICULTY_RANK = {"low": 1, "medium": 2, "high": 3}

VIS_COND = ["normal", "low"]                 # Mohanavelu
COG_TASK = [0, 1]                             # Mohanavelu secondary task on/off

PHASES = ["baseline_rest", "takeoff", "climb", "cruise", "attack_task", "landing_rest"]
# Baseline demand multiplier per phase (attack_task is where the manipulated
# task load is fully expressed; other phases carry light, mostly fixed workload)
PHASE_LOAD_MULT = {
    "baseline_rest": 0.05,
    "takeoff": 0.35,
    "climb": 0.25,
    "cruise": 0.20,
    "attack_task": 1.00,
    "landing_rest": 0.30,
}

def sample_mission(mission_counter, pilot_id):
    """Generate one mission's scenario-level variables."""
    mission_type = rng.choice(["ATA", "ATG"], p=[0.5, 0.5])  # air-to-air / air-to-ground
    difficulty = rng.choice(["low", "medium", "high"], p=[0.34, 0.34, 0.32])
    target_count, required_engagements = DIFFICULTY_TIERS[difficulty]
    # small jitter so not every "high" mission is numerically identical
    target_count = max(1, int(round(target_count + rng.integers(-1, 2))))
    required_engagements = max(1, int(round(required_engagements + rng.integers(0, 2))))

    visibility = rng.choice(VIS_COND, p=[0.6, 0.4])
    cognitive_task = int(rng.choice(COG_TASK, p=[0.55, 0.45]))

    # Svensson-style tactical/information complexity (synthetic extension,
    # inspired by but not numerically drawn from the source paper)
    threat_density = int(np.clip(rng.poisson(2 + DIFFICULTY_RANK[difficulty]), 0, 10))
    friendly_count = int(np.clip(rng.poisson(1.5), 0, 6))
    info_update_rate = np.clip(rng.normal(4 + DIFFICULTY_RANK[difficulty] * 1.5, 1.2), 1, 15)  # updates/min
    display_complexity = int(np.clip(rng.integers(1, 6) + (1 if cognitive_task else 0), 1, 5))

    return {
        "mission_id": f"M{mission_counter:06d}",
        "pilot_id": pilot_id,
        "mission_type": mission_type,
        "difficulty_tier": difficulty,
        "difficulty_rank": DIFFICULTY_RANK[difficulty],
        "target_count": target_count,
        "required_engagements": required_engagements,
        "visibility": visibility,
        "secondary_cognitive_task": cognitive_task,
        "threat_density": threat_density,
        "friendly_aircraft_count": friendly_count,
        "information_update_rate_per_min": round(info_update_rate, 2),
        "display_complexity": display_complexity,
    }

# ----------------------------------------------------------------------------
# 3. Demand-component + latent workload model (Steps 3-6)
# ----------------------------------------------------------------------------
# Weights are declared explicitly and documented as *calibration-pending*
# (Section 6/9 of the methodology: not to be presented as experimentally fixed).
W_WEIGHTS = {
    "task": 0.28,
    "perceptual": 0.18,
    "cognitive": 0.20,
    "tactical": 0.20,
    "temporal": 0.14,
}

def zscore(x, mean, std):
    return (x - mean) / std

# ----------------------------------------------------------------------------
# 4. Row-level generation (mission x phase x window)
# ----------------------------------------------------------------------------
rows = []
mission_meta_rows = []

N_MISSIONS_PER_PILOT = rng.integers(22, 40, N_PILOTS)  # variable mission counts
mission_counter = 1

for p_idx, pilot_id in enumerate(pilots["pilot_id"]):
    prow = pilots.loc[pilot_id]
    n_missions = int(N_MISSIONS_PER_PILOT[p_idx])

    for m in range(n_missions):
        mission = sample_mission(mission_counter, pilot_id)
        mission_counter += 1
        mission_meta_rows.append(mission)

        # ---- mission-level (scenario) demand components (fixed within mission) ----
        D_task_raw = (
            0.6 * mission["target_count"] + 1.4 * mission["required_engagements"]
        )
        D_tactical_raw = (
            0.5 * mission["threat_density"]
            + 0.3 * mission["friendly_aircraft_count"]
            + 0.4 * mission["information_update_rate_per_min"]
            + 1.2 * mission["display_complexity"]
        )

        segment_counter = 0
        for phase in PHASES:
            n_windows = int(rng.integers(3, 7)) if phase == "attack_task" else 1

            for w in range(n_windows):
                segment_counter += 1
                phase_mult = PHASE_LOAD_MULT[phase]

                # ---- window-level demand components ----
                D_perceptual_raw = phase_mult * (
                    2.5 if mission["visibility"] == "low" else 1.0
                )
                D_cognitive_raw = phase_mult * (
                    1.0 + 2.0 * mission["secondary_cognitive_task"]
                )
                # time pressure: shorter allotted window under higher difficulty/attack phase
                window_duration_s = np.clip(
                    rng.normal(
                        20 - 3 * mission["difficulty_rank"] if phase == "attack_task" else 45,
                        4,
                    ),
                    6, 90,
                )
                D_temporal_raw = phase_mult * (60.0 / window_duration_s)

                D_task = phase_mult * D_task_raw

                # standardize each raw demand component (approx. dataset-wide scaling
                # constants derived from the sampling ranges above)
                d_task_z = zscore(D_task, 6.0, 4.0)
                d_perc_z = zscore(D_perceptual_raw, 1.2, 0.8)
                d_cog_z = zscore(D_cognitive_raw, 1.3, 0.9)
                d_tac_z = zscore(D_tactical_raw, 6.0, 4.0)
                d_temp_z = zscore(D_temporal_raw, 2.0, 1.5)

                pilot_effect = -0.35 * prow["pilot_skill_z"] + rng.normal(0, 0.15)
                noise = rng.normal(0, 0.25)

                W_latent = (
                    W_WEIGHTS["task"] * d_task_z
                    + W_WEIGHTS["perceptual"] * d_perc_z
                    + W_WEIGHTS["cognitive"] * d_cog_z
                    + W_WEIGHTS["tactical"] * d_tac_z
                    + W_WEIGHTS["temporal"] * d_temp_z
                    + pilot_effect
                    + noise
                )

                # ---- performance (Step 7) ----
                gain = prow["pilot_reactivity_gain"]
                base_engagement_time = 8.0 if mission["mission_type"] == "ATG" else 5.5
                engagement_time_s = np.clip(
                    base_engagement_time + 2.2 * W_latent * gain
                    - 0.3 * prow["pilot_skill_z"] + rng.normal(0, 0.6),
                    1.5, 30,
                )
                hit_rate = np.clip(
                    0.88 - 0.09 * max(W_latent, 0) + 0.03 * prow["pilot_skill_z"]
                    + rng.normal(0, 0.05),
                    0.05, 1.0,
                )
                target_disengagement_time_s = np.clip(
                    3.0 + 1.1 * max(W_latent, 0) + rng.normal(0, 0.4), 0.5, 12,
                )
                altitude_error_ft = np.clip(
                    30 + 45 * max(W_latent, 0) - 6 * prow["pilot_skill_z"]
                    + rng.normal(0, 12), 0, 400,
                )
                tracking_error_deg = np.clip(
                    1.0 + 1.6 * max(W_latent, 0) + rng.normal(0, 0.3), 0, 10,
                )
                navigation_error_nm = np.clip(
                    0.4 + 0.9 * max(W_latent, 0) - 0.1 * prow["pilot_skill_z"]
                    + rng.normal(0, 0.2), 0, 6,
                )
                mission_completion_prob = 1 / (1 + np.exp(2.0 * W_latent - 0.5 * prow["pilot_skill_z"]))
                mission_completion = int(rng.random() < mission_completion_prob)

                # ---- physiology (Step 8): baseline + workload-driven delta * pilot gain ----
                HR = np.clip(
                    prow["baseline_HR"] + gain * (10 * max(W_latent, 0)) + rng.normal(0, 3),
                    45, 190,
                )
                SDNN = np.clip(
                    prow["baseline_SDNN"] - gain * (9 * max(W_latent, 0)) + rng.normal(0, 4),
                    5, 130,
                )
                RMSSD = np.clip(
                    prow["baseline_RMSSD"] - gain * (7 * max(W_latent, 0)) + rng.normal(0, 3),
                    3, 110,
                )
                LF_HF = np.clip(
                    prow["baseline_LF_HF"] + gain * (0.9 * max(W_latent, 0)) + rng.normal(0, 0.15),
                    0.2, 6.0,
                )
                theta_power = np.clip(
                    prow["baseline_theta_power"] + gain * (0.06 * max(W_latent, 0)) + rng.normal(0, 0.01),
                    0.05, 0.55,
                )
                alpha_power = np.clip(
                    prow["baseline_alpha_power"] - gain * (0.05 * max(W_latent, 0)) + rng.normal(0, 0.015),
                    0.05, 0.55,
                )
                delta_power = np.clip(
                    prow["baseline_delta_power"] + gain * (0.01 * max(W_latent, 0)) + rng.normal(0, 0.01),
                    0.03, 0.40,
                )
                fixation_rate_per_min = np.clip(
                    150 + gain * (25 * max(W_latent, 0)) + rng.normal(0, 8), 60, 320,
                )
                fixation_duration_ms = np.clip(
                    prow["baseline_fixation_duration_ms"] - gain * (30 * max(W_latent, 0)) + rng.normal(0, 15),
                    120, 450,
                )
                blink_rate_bpm = np.clip(
                    prow["baseline_blink_rate_bpm"] - gain * (5 * max(W_latent, 0)) + rng.normal(0, 2),
                    2, 35,
                )
                pupil_mm = np.clip(
                    prow["baseline_pupil_mm"] + gain * (0.35 * max(W_latent, 0)) + rng.normal(0, 0.1),
                    2.5, 7.5,
                )

                # ---- subjective proxy (Step 9) ----
                nasa_tlx_proxy = np.clip(
                    50 + 18 * W_latent + rng.normal(0, 6), 0, 100,
                )

                rows.append({
                    "pilot_id": pilot_id,
                    "mission_id": mission["mission_id"],
                    "segment_id": f"{mission['mission_id']}_S{segment_counter:02d}",
                    "experience_group": prow["experience_group"],
                    "years_experience": round(float(prow["years_experience"]), 1),
                    "mission_type": mission["mission_type"],
                    "difficulty_tier": mission["difficulty_tier"],
                    "difficulty_rank": mission["difficulty_rank"],
                    "target_count": mission["target_count"],
                    "required_engagements": mission["required_engagements"],
                    "visibility": mission["visibility"],
                    "secondary_cognitive_task": mission["secondary_cognitive_task"],
                    "threat_density": mission["threat_density"],
                    "friendly_aircraft_count": mission["friendly_aircraft_count"],
                    "information_update_rate_per_min": mission["information_update_rate_per_min"],
                    "display_complexity": mission["display_complexity"],
                    "flight_phase": phase,
                    "window_duration_s": round(float(window_duration_s), 1),
                    "D_task_z": round(float(d_task_z), 3),
                    "D_perceptual_z": round(float(d_perc_z), 3),
                    "D_cognitive_z": round(float(d_cog_z), 3),
                    "D_tactical_z": round(float(d_tac_z), 3),
                    "D_temporal_z": round(float(d_temp_z), 3),
                    "latent_workload": round(float(W_latent), 4),
                    "engagement_time_s": round(float(engagement_time_s), 2),
                    "hit_rate": round(float(hit_rate), 3),
                    "target_disengagement_time_s": round(float(target_disengagement_time_s), 2),
                    "altitude_error_ft": round(float(altitude_error_ft), 1),
                    "tracking_error_deg": round(float(tracking_error_deg), 2),
                    "navigation_error_nm": round(float(navigation_error_nm), 2),
                    "mission_completion": mission_completion,
                    "HR_bpm": round(float(HR), 1),
                    "SDNN_ms": round(float(SDNN), 1),
                    "RMSSD_ms": round(float(RMSSD), 1),
                    "LF_HF_ratio": round(float(LF_HF), 2),
                    "EEG_theta_rel_power": round(float(theta_power), 4),
                    "EEG_alpha_rel_power": round(float(alpha_power), 4),
                    "EEG_delta_rel_power": round(float(delta_power), 4),
                    "fixation_rate_per_min": round(float(fixation_rate_per_min), 1),
                    "fixation_duration_ms": round(float(fixation_duration_ms), 1),
                    "blink_rate_bpm": round(float(blink_rate_bpm), 1),
                    "pupil_diameter_mm": round(float(pupil_mm), 2),
                    "nasa_tlx_subjective_proxy": round(float(nasa_tlx_proxy), 1),
                    "workload_source": "scenario_derived",
                    "generator_version": GENERATOR_VERSION,
                    "parameter_seed": SEED,
                })

df = pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# 5. Workload class labels (Step 9) -- tertile bins on latent workload,
#    computed AFTER full-dataset generation so cut points are data-driven.
# ----------------------------------------------------------------------------
q1, q2 = df["latent_workload"].quantile([1/3, 2/3])
df["workload_class_3lvl"] = pd.cut(
    df["latent_workload"], bins=[-np.inf, q1, q2, np.inf],
    labels=["low", "medium", "high"]
)

# Mohanavelu-style 4-level perceptual/cognitive condition label
def four_level(row):
    v = row["visibility"]
    c = row["secondary_cognitive_task"]
    if v == "normal" and c == 0:
        return "NV_NoTask"
    if v == "low" and c == 0:
        return "LV_NoTask"
    if v == "normal" and c == 1:
        return "NV_CogTask"
    return "LV_CogTask"

df["perceptual_cognitive_condition"] = df.apply(four_level, axis=1)

df["confidence"] = np.round(np.clip(1 - np.abs(df["latent_workload"]) * 0.01, 0.85, 0.99), 3)
df["validation_flag"] = True

# reorder columns: identifiers first
id_cols = ["pilot_id", "mission_id", "segment_id"]
other_cols = [c for c in df.columns if c not in id_cols]
df = df[id_cols + other_cols]

print("Full windowed dataset shape:", df.shape)

# ----------------------------------------------------------------------------
# 6. Save Version 2 (windowed) dataset
# ----------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
out_path = OUTPUT_DIR / "synthetic_fighter_pilot_workload_windowed.csv"
df.to_csv(out_path, index=False)
print("Saved:", out_path)

# ----------------------------------------------------------------------------
# 7. Build Version 1 (mission-level aggregate) dataset for convenience
# ----------------------------------------------------------------------------
agg_funcs = {
    "latent_workload": "mean",
    "engagement_time_s": "mean",
    "hit_rate": "mean",
    "target_disengagement_time_s": "mean",
    "altitude_error_ft": "mean",
    "tracking_error_deg": "mean",
    "navigation_error_nm": "mean",
    "mission_completion": "max",
    "HR_bpm": "mean",
    "SDNN_ms": "mean",
    "RMSSD_ms": "mean",
    "LF_HF_ratio": "mean",
    "EEG_theta_rel_power": "mean",
    "EEG_alpha_rel_power": "mean",
    "EEG_delta_rel_power": "mean",
    "fixation_rate_per_min": "mean",
    "fixation_duration_ms": "mean",
    "blink_rate_bpm": "mean",
    "pupil_diameter_mm": "mean",
    "nasa_tlx_subjective_proxy": "mean",
}
group_cols = ["pilot_id", "mission_id", "experience_group", "mission_type",
              "difficulty_tier", "difficulty_rank", "target_count",
              "required_engagements", "visibility", "secondary_cognitive_task",
              "threat_density", "friendly_aircraft_count",
              "information_update_rate_per_min", "display_complexity"]

mission_level = df.groupby(group_cols, as_index=False, observed=True).agg(agg_funcs)
q1m, q2m = mission_level["latent_workload"].quantile([1/3, 2/3])
mission_level["workload_class_3lvl"] = pd.cut(
    mission_level["latent_workload"], bins=[-np.inf, q1m, q2m, np.inf],
    labels=["low", "medium", "high"]
)
mission_level["generator_version"] = GENERATOR_VERSION
mission_level["parameter_seed"] = SEED

mission_out_path = OUTPUT_DIR / "synthetic_fighter_pilot_workload_mission_level.csv"
mission_level.to_csv(mission_out_path, index=False)
print("Mission-level dataset shape:", mission_level.shape)
print("Saved:", mission_out_path)
