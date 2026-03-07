from dataclasses import dataclass
from fastai.tabular.all import load_learner
from itertools import product
from pathlib import Path
from pybaseball import batting_stats, playerid_lookup, playerid_reverse_lookup, statcast_batter, statcast_pitcher
from pybaseball.cache import cache

import pandas as pd
import numpy as np


cache.enable()


MODELS_DIR = Path("models")
SWING_MODEL_PATH = MODELS_DIR / "swing_model.pkl"
BSX_MODEL_PATH = MODELS_DIR / "bsx_model.pkl"

swing_model = load_learner(SWING_MODEL_PATH, cpu=False)
bsx_model = load_learner(BSX_MODEL_PATH, cpu=False)


@dataclass
class SwingFeatures:
    p_throws: str
    stand: str
    inning_topbot: str
    prev_pitch_type: str | None
    prev_type: str | None
    pitch_type: str
    inning: int
    bat_score: int
    fld_score: int
    outs_when_up: int
    on_3b: bool
    on_2b: bool
    on_1b: bool
    balls: int
    strikes: int
    release_speed: float
    pfx_x: float
    pfx_z: float
    K_rate: float
    BB_rate: float
    px: float
    pz: float
    prev_px: float | None
    prev_pz: float | None

    def to_dict(self) -> dict:
        has_prev_px = float(pd.notna(self.prev_px))
        has_prev_pz = float(pd.notna(self.prev_pz))
        prev_px = float(self.prev_px) if has_prev_px else 0.0
        prev_pz = float(self.prev_pz) if has_prev_pz else 0.0

        return {
            "p_throws": self.p_throws,
            "stand": self.stand,
            "inning_topbot": self.inning_topbot,
            "prev_pitch_type": self.prev_pitch_type,
            "prev_type": self.prev_type,
            "pitch_type": self.pitch_type,
            "inning": self.inning,
            "bat_score": self.bat_score,
            "fld_score": self.fld_score,
            "outs_when_up": self.outs_when_up,
            "on_3b": self.on_3b,
            "on_2b": self.on_2b,
            "on_1b": self.on_1b,
            "balls": self.balls,
            "strikes": self.strikes,
            "release_speed": self.release_speed,
            "pfx_x": self.pfx_x,
            "pfx_z": self.pfx_z,
            "K%": self.K_rate,
            "BB%": self.BB_rate,
            "px": self.px,
            "pz": self.pz,
            "prev_px": prev_px,
            "prev_pz": prev_pz,
            "has_prev_px": has_prev_px,
            "has_prev_pz": has_prev_pz,
        }

@dataclass
class BSXFeatures:
    p_throws: str
    stand: str
    pitch_type: str
    release_speed: float
    pfx_x: float
    pfx_z: float
    px: float
    pz: float
    swing: bool

    def to_dict(self) -> dict:
        return {
            "p_throws": self.p_throws,
            "stand": self.stand,
            "pitch_type": self.pitch_type,
            "release_speed": self.release_speed,
            "pfx_x": self.pfx_x,
            "pfx_z": self.pfx_z,
            "px": self.px,
            "pz": self.pz,
            "swing": self.swing,
        }


@dataclass
class SequenceStep:
    state_before: tuple[int, int]
    recommended_pitch: tuple[str, float, float]
    outcome_probs: dict[str, float]
    predicted_outcome: str
    state_after: tuple[int, int] | None
    terminal_outcome: str | None
    strikeout_value_from_state: float

    def to_dict(self) -> dict:
        return {
            "state_before": self.state_before,
            "recommended_pitch": self.recommended_pitch,
            "outcome_probs": self.outcome_probs,
            "predicted_outcome": self.predicted_outcome,
            "state_after": self.state_after,
            "terminal_outcome": self.terminal_outcome,
            "strikeout_value_from_state": self.strikeout_value_from_state,
        }


def get_pitcher_features(pitcher_name: str, season: int = 2025) -> dict[str, float | str]:
    name_parts = pitcher_name.strip().split()
    if len(name_parts) < 2:
        raise ValueError("pitcher_name must include first and last name.")

    first_name = " ".join(name_parts[:-1])
    last_name = name_parts[-1]

    lookup = playerid_lookup(last_name, first_name)
    if lookup.empty:
        raise ValueError(f"No pitcher found for name: {pitcher_name}")

    pitcher_id_mlbam = int(lookup.iloc[0]["key_mlbam"])
    start_date = f"{season}-03-01"
    end_date = f"{season}-10-15"
    pitcher_pitches = statcast_pitcher(start_date, end_date, pitcher_id_mlbam)

    if pitcher_pitches.empty:
        raise ValueError(f"No Statcast data found for {pitcher_name} in {season}.")

    p_throws_series = pitcher_pitches["p_throws"].dropna()
    if p_throws_series.empty:
        raise ValueError(f"Missing throwing-hand data for {pitcher_name}.")

    return {
        "p_throws": str(p_throws_series.mode().iloc[0]).upper(),
        "release_speed": float(pitcher_pitches["release_speed"].dropna().mean()),
        "pfx_x": float(pitcher_pitches["pfx_x"].dropna().mean()),
        "pfx_z": float(pitcher_pitches["pfx_z"].dropna().mean()),
    }


def get_batter_features(batter_name: str, season: int = 2025) -> dict[str, float | str]:
    name_parts = batter_name.strip().split()
    if len(name_parts) < 2:
        raise ValueError("batter_name must include first and last name.")

    first_name = " ".join(name_parts[:-1])
    last_name = name_parts[-1]

    lookup = playerid_lookup(last_name, first_name)
    if lookup.empty:
        raise ValueError(f"No batter found for name: {batter_name}")

    batter_id_mlbam = int(lookup.iloc[0]["key_mlbam"])
    reverse_lookup = playerid_reverse_lookup([batter_id_mlbam], key_type="mlbam")
    if reverse_lookup.empty or pd.isna(reverse_lookup.iloc[0]["key_fangraphs"]):
        raise ValueError(f"No FanGraphs ID found for batter: {batter_name}")

    batter_id_fg = int(reverse_lookup.iloc[0]["key_fangraphs"])
    start_date = f"{season}-03-01"
    end_date = f"{season}-10-15"
    batter_pitches = statcast_batter(start_date, end_date, batter_id_mlbam)

    if batter_pitches.empty:
        raise ValueError(f"No Statcast data found for {batter_name} in {season}.")

    stand_series = batter_pitches["stand"].dropna()
    if stand_series.empty:
        raise ValueError(f"Missing batting-side data for {batter_name}.")

    batter_stats = batting_stats(season, qual=0)
    batter_stats = batter_stats[batter_stats["IDfg"] == batter_id_fg]
    if batter_stats.empty:
        raise ValueError(f"No batting stats found for {batter_name} in {season}.")

    return {
        "stand": str(stand_series.mode().iloc[0]).upper(),
        "K_rate": float(batter_stats["K%"].iloc[0]),
        "BB_rate": float(batter_stats["BB%"].iloc[0]),
    }


def predict_swing(features: SwingFeatures) -> dict[bool, float]:
    sample_df = pd.DataFrame([features.to_dict()])
    _, _, pred_probs = swing_model.predict(sample_df.iloc[0])

    return {
        False: float(pred_probs[0]),
        True: float(pred_probs[1]),
    }

def predict_bsx(features: BSXFeatures) -> dict[str, float]:
    sample_df = pd.DataFrame([features.to_dict()])
    _, _, pred_probs = bsx_model.predict(sample_df.iloc[0])

    return {
        "B": float(pred_probs[0]),
        "S": float(pred_probs[1]),
        "X": float(pred_probs[2]),
    }


def predict_swing_batch(features_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dl = swing_model.dls.test_dl(features_df)
    pred_output = swing_model.get_preds(dl=dl)
    pred_probs = pred_output[0]
    pred_probs_df = pd.DataFrame(pred_probs.cpu().numpy(), columns=["p_false", "p_true"])
    return pred_probs_df["p_false"], pred_probs_df["p_true"]


def predict_bsx_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    dl = bsx_model.dls.test_dl(features_df)
    pred_output = bsx_model.get_preds(dl=dl)
    pred_probs = pred_output[0]
    return pd.DataFrame(pred_probs.cpu().numpy(), columns=["p_B", "p_S", "p_X"])


def pitch_outcome_probs(
    p_throws: str,
    stand: str,
    inning_topbot: str,
    prev_pitch_type: str | None,
    prev_type: str | None,
    inning: int,
    bat_score: int,
    fld_score: int,
    outs_when_up: int,
    on_3b: bool,
    on_2b: bool,
    on_1b: bool,
    balls: int,
    strikes: int,
    release_speed: float,
    pfx_x: float,
    pfx_z: float,
    K_rate: float,
    BB_rate: float,
    prev_px: float | None,
    prev_pz: float | None,
) -> dict[tuple[str, float, float], dict[str, tuple[float, float, float]]]:
    pitch_types = ["ff", "ch", "sl", "si", "fc", "cu", "kc", "st", "fs", "sv", "fa", "ep", "cs", "sc", "kn", "po", "fo", "un"]
    px_values = np.linspace(-2.0, 2.0, 20)
    pz_values = np.linspace(-1.5, 2.5, 20)
    grid = list(product(pitch_types, px_values, pz_values))

    swing_rows: list[dict] = []
    bsx_rows_swing: list[dict] = []
    bsx_rows_no_swing: list[dict] = []

    for pitch_type, px, pz in grid:
        swing_rows.append(
            SwingFeatures(
                p_throws=p_throws,
                stand=stand,
                inning_topbot=inning_topbot,
                prev_pitch_type=prev_pitch_type,
                prev_type=prev_type,
                pitch_type=pitch_type,
                inning=inning,
                bat_score=bat_score,
                fld_score=fld_score,
                outs_when_up=outs_when_up,
                on_3b=on_3b,
                on_2b=on_2b,
                on_1b=on_1b,
                balls=balls,
                strikes=strikes,
                release_speed=release_speed,
                pfx_x=pfx_x,
                pfx_z=pfx_z,
                K_rate=K_rate,
                BB_rate=BB_rate,
                px=px,
                pz=pz,
                prev_px=prev_px,
                prev_pz=prev_pz,
            ).to_dict()
        )
        bsx_rows_swing.append(
            BSXFeatures(
                p_throws=p_throws,
                stand=stand,
                pitch_type=pitch_type,
                release_speed=release_speed,
                pfx_x=pfx_x,
                pfx_z=pfx_z,
                px=px,
                pz=pz,
                swing=True,
            ).to_dict()
        )
        bsx_rows_no_swing.append(
            BSXFeatures(
                p_throws=p_throws,
                stand=stand,
                pitch_type=pitch_type,
                release_speed=release_speed,
                pfx_x=pfx_x,
                pfx_z=pfx_z,
                px=px,
                pz=pz,
                swing=False,
            ).to_dict()
        )

    swing_df = pd.DataFrame(swing_rows)
    bsx_df_swing = pd.DataFrame(bsx_rows_swing)
    bsx_df_no_swing = pd.DataFrame(bsx_rows_no_swing)

    swing_false_probs, swing_true_probs = predict_swing_batch(swing_df)
    bsx_probs_swing = predict_bsx_batch(bsx_df_swing)
    bsx_probs_no_swing = predict_bsx_batch(bsx_df_no_swing)

    p_B = bsx_probs_swing["p_B"] * swing_true_probs + bsx_probs_no_swing["p_B"] * swing_false_probs
    p_S = bsx_probs_swing["p_S"] * swing_true_probs + bsx_probs_no_swing["p_S"] * swing_false_probs
    p_X = bsx_probs_swing["p_X"] * swing_true_probs + bsx_probs_no_swing["p_X"] * swing_false_probs

    results = {}
    for i, (pitch_type, px, pz) in enumerate(grid):
        results[pitch_type, px, pz] = {
            "B": float(p_B.iloc[i]),
            "S": float(p_S.iloc[i]),
            "X": float(p_X.iloc[i]),
        }
    return results


def _count_state_is_terminal(balls: int, strikes: int) -> bool:
    return balls >= 4 or strikes >= 3


def _next_count_state_from_outcome(
    balls: int,
    strikes: int,
    outcome: str,
) -> tuple[tuple[int, int] | None, float, str | None]:
    if outcome == "X":
        return None, 0.0, "X"
    if outcome == "B":
        if balls >= 3:
            return None, 0.0, "BB"
        return (balls + 1, strikes), 0.0, None
    if outcome == "S":
        if strikes >= 2:
            return None, 1.0, "K"
        return (balls, strikes + 1), 0.0, None
    raise ValueError(f"Unsupported outcome: {outcome}")


def _state_key(state: tuple[int, int]) -> str:
    balls, strikes = state
    return f"{balls}-{strikes}"


def optimal_pitch_sequence(
    p_throws: str,
    stand: str,
    inning_topbot: str,
    prev_pitch_type: str | None,
    prev_type: str | None,
    inning: int,
    bat_score: int,
    fld_score: int,
    outs_when_up: int,
    on_3b: bool,
    on_2b: bool,
    on_1b: bool,
    release_speed: float,
    pfx_x: float,
    pfx_z: float,
    K_rate: float,
    BB_rate: float,
    prev_px: float | None,
    prev_pz: float | None,
    initial_balls: int = 0,
    initial_strikes: int = 0,
    max_steps: int = 12,
) -> dict:
    nonterminal_states = [(balls, strikes) for balls in range(4) for strikes in range(3)]
    state_values: dict[tuple[int, int], float] = {(balls, strikes): 0.0 for balls, strikes in nonterminal_states}
    best_actions: dict[tuple[int, int], tuple[str, float, float]] = {}
    best_action_outcomes: dict[tuple[int, int], dict[str, float]] = {}

    for balls in reversed(range(4)):
        for strikes in reversed(range(3)):
            outcome_grid = pitch_outcome_probs(
                p_throws=p_throws,
                stand=stand,
                inning_topbot=inning_topbot,
                prev_pitch_type=prev_pitch_type,
                prev_type=prev_type,
                inning=inning,
                bat_score=bat_score,
                fld_score=fld_score,
                outs_when_up=outs_when_up,
                on_3b=on_3b,
                on_2b=on_2b,
                on_1b=on_1b,
                balls=balls,
                strikes=strikes,
                release_speed=release_speed,
                pfx_x=pfx_x,
                pfx_z=pfx_z,
                K_rate=K_rate,
                BB_rate=BB_rate,
                prev_px=prev_px,
                prev_pz=prev_pz,
            )

            best_value = -1.0
            best_pitch: tuple[str, float, float] | None = None
            best_probs: dict[str, float] | None = None

            for pitch, probs in outcome_grid.items():
                expected_k_prob = 0.0
                for outcome, p_outcome in probs.items():
                    next_state, terminal_k_prob, _ = _next_count_state_from_outcome(
                        balls=balls,
                        strikes=strikes,
                        outcome=outcome,
                    )
                    if next_state is None:
                        expected_k_prob += p_outcome * terminal_k_prob
                    else:
                        expected_k_prob += p_outcome * state_values[next_state]

                if expected_k_prob > best_value:
                    best_value = expected_k_prob
                    best_pitch = (pitch[0], float(pitch[1]), float(pitch[2]))
                    best_probs = probs

            if best_pitch is None or best_probs is None:
                raise RuntimeError(f"No valid pitch found for state ({balls}, {strikes}).")

            state_values[balls, strikes] = float(best_value)
            best_actions[balls, strikes] = best_pitch
            best_action_outcomes[balls, strikes] = best_probs

    current_state = (initial_balls, initial_strikes)
    if _count_state_is_terminal(*current_state):
        raise ValueError(f"Initial count is already terminal: {current_state}")
    if current_state not in state_values:
        raise ValueError(f"Initial count out of supported range: {current_state}")

    sequence_steps: list[SequenceStep] = []
    terminal_outcome: str | None = None

    for _ in range(max_steps):
        if current_state not in best_actions:
            break

        pitch = best_actions[current_state]
        outcome_probs = best_action_outcomes[current_state]
        predicted_outcome = max(outcome_probs, key=outcome_probs.get)
        next_state, _, terminal_label = _next_count_state_from_outcome(
            balls=current_state[0],
            strikes=current_state[1],
            outcome=predicted_outcome,
        )
        strikeout_value = state_values[current_state]

        step = SequenceStep(
            state_before=current_state,
            recommended_pitch=pitch,
            outcome_probs={k: float(v) for k, v in outcome_probs.items()},
            predicted_outcome=predicted_outcome,
            state_after=next_state,
            terminal_outcome=terminal_label,
            strikeout_value_from_state=strikeout_value,
        )
        sequence_steps.append(step)

        if terminal_label is not None:
            terminal_outcome = terminal_label
            break
        if next_state is None:
            break
        current_state = next_state

    state_values_pretty = {_state_key(state): value for state, value in sorted(state_values.items())}
    initial_state = (initial_balls, initial_strikes)
    first_pitch = best_actions[initial_state]

    return {
        "best_first_pitch": first_pitch,
        "expected_strikeout_probability": state_values[initial_state],
        "sequence": [step.to_dict() for step in sequence_steps],
        "state_values": state_values_pretty,
        "terminal_outcome_along_recommended_path": terminal_outcome,
    }


PITCH_TYPES = {
    "ff": "four-seam fastball",
    "si": "sinker",
    "ft": "two-seam fastball",
    "fc": "cutter",
    "ch": "changeup",
    "sl": "slider",
    "st": "sweeper",
    "sv": "slurve",
    "cu": "curveball",
    "kc": "knuckle-curve",
    "fs": "splitter",
    "fo": "forkball",
    "sc": "screwball",
    "gy": "gyroball",
    "ep": "eephus",
    "kn": "knuckleball",
    "in": "intentional ball",
    "po": "pitchout",
    "ab": "automatic ball",
    "as": "automatic strike",
    "np": "no pitch",
    "un": "unknown",
}


if __name__ == "__main__":
    pitcher_features = get_pitcher_features("Shohei Ohtani", 2025)
    batter_features = get_batter_features("George Springer", 2025)

    recommendation = optimal_pitch_sequence(
        p_throws=pitcher_features["p_throws"],
        stand=batter_features["stand"],
        inning_topbot="Top",
        prev_pitch_type=None,
        prev_type=None,
        inning=0,
        bat_score=0,
        fld_score=0,
        outs_when_up=0,
        on_3b=True,
        on_2b=False,
        on_1b=False,
        release_speed=float(pitcher_features["release_speed"]),
        pfx_x=float(pitcher_features["pfx_x"]),
        pfx_z=float(pitcher_features["pfx_z"]),
        K_rate=float(batter_features["K_rate"]),
        BB_rate=float(batter_features["BB_rate"]),
        prev_px=None,
        prev_pz=None,
        initial_balls=0,
        initial_strikes=0,
    )

    print(f"Expected strikeout probability: {recommendation['expected_strikeout_probability']:.4f}")
    print("\nRecommended path:")
    for i, step in enumerate(recommendation["sequence"], start=1):
        pitch_type, px, pz = step["recommended_pitch"]
        pitch_type_name = PITCH_TYPES.get(pitch_type, pitch_type)
        print(f"{i}. Count: {step['state_before']}")
        print(f"\tpitch: {pitch_type_name} at {px:.2f}, {pz:.2f}")
        print(f"\tpredicted outcome: {step['predicted_outcome']}")
        print(f"\tnext count: {step['state_after']}")
        print(f"\tvalue: {step['strikeout_value_from_state']:.4f}")
        print()
