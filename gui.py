from dataclasses import dataclass
from main import BSXFeatures, SwingFeatures, predict_bsx_batch, predict_swing_batch
from nicegui import ui

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PITCH_TYPES = {
    "Four-seam fastball": "ff",
    "Changeup": "ch",
    "Slider": "sl",
    "Sinker": "si",
    "Cutter": "fc",
    "Curveball": "cu",
    "Knuckle-curve": "kc",
    "Sweeper": "st",
    "Splitter": "fs",
    "Slurve": "sv",
    "Fastball": "fa",
    "Eephus": "ep",
    "Slow curve": "cs",
    "Screwball": "sc",
    "Knuckleball": "kn",
    "Pitchout": "po",
    "Forkball": "fo",
}

HANDEDNESS_OPTIONS = ["L", "R"]
INNING_HALVES = ["Top", "Bot"]

PLATE_HEIGHT = 3.0
X_AXIS_RANGE = (-1.35, 1.35)
Y_AXIS_RANGE = (-0.35, PLATE_HEIGHT + 0.35)

give = 1
num_points = 70
xs = np.linspace(-1 - give, 1 + give, num_points)
zs = np.linspace(Y_AXIS_RANGE[0] / PLATE_HEIGHT, Y_AXIS_RANGE[1] / PLATE_HEIGHT, num_points)


@dataclass
class AppState:
    release_speed: float = 90
    pitch_type_name: str = next(iter(PITCH_TYPES))
    p_throws: str = "L"
    stand: str = "R"
    inning_topbot: str = "Top"
    inning: int = 9
    bat_score: int = 0
    fld_score: int = 0
    outs_when_up: int = 0
    on_3b: bool = True
    on_2b: bool = True
    on_1b: bool = True
    pfx_x: float = 0.0
    pfx_z: float = 0.0
    k_rate: float = 0.236
    bb_rate: float = 0.183
    prev_pitch_type_name: str | None = None
    prev_type: str | None = None
    prev_pitch_x: float | None = None
    prev_pitch_z: float | None = None
    pitch_x: float | None = None
    pitch_z: float | None = None
    balls: int = 0
    strikes: int = 0
    swing_features: SwingFeatures | None = None
    selected_metric: str = "S"
    optimization_direction: str = "max"
    outcome_probs: pd.DataFrame | None = None


METRIC_COLUMNS = {
    "S": "p_S",
    "B": "p_B",
    "X": "p_X",
    "swing": "p_swing",
    "take": "p_take",
}

METRIC_LABELS = {
    "S": "P(strike)",
    "B": "P(ball)",
    "X": "P(in-play)",
    "swing": "P(swing)",
    "take": "P(take)",
}

METRIC_OPTIONS = {
    "S": "Strike",
    "B": "Ball",
    "X": "In-play",
    "swing": "Swing",
    "take": "Take",
}

OPTIMIZATION_DIRECTIONS = {
    "max": "Maximize selected metric",
    "min": "Minimize selected metric",
}

PREV_PITCH_RESULT_OPTIONS = {
    "none": "None",
    "B": "Ball",
    "S": "Strike",
    "X": "In-play",
}


def make_swing_grid_features(state: AppState, pitch_type_name: str | None = None) -> pd.DataFrame:
    selected_pitch_type = pitch_type_name or state.pitch_type_name
    rows: list[dict] = []
    for x in xs:
        for z in zs:
            rows.append(
                SwingFeatures(
                    p_throws=state.p_throws,
                    stand=state.stand,
                    inning_topbot=state.inning_topbot,
                    prev_pitch_type=(
                        PITCH_TYPES[state.prev_pitch_type_name] if state.prev_pitch_type_name is not None else None
                    ),
                    prev_type=state.prev_type,
                    pitch_type=PITCH_TYPES[selected_pitch_type],
                    inning=state.inning,
                    bat_score=state.bat_score,
                    fld_score=state.fld_score,
                    outs_when_up=state.outs_when_up,
                    on_3b=state.on_3b,
                    on_2b=state.on_2b,
                    on_1b=state.on_1b,
                    balls=state.balls,
                    strikes=state.strikes,
                    release_speed=state.release_speed,
                    pfx_x=state.pfx_x,
                    pfx_z=state.pfx_z,
                    K_rate=state.k_rate,
                    BB_rate=state.bb_rate,
                    px=float(x),
                    pz=float(z),
                    prev_px=state.prev_pitch_x,
                    prev_pz=state.prev_pitch_z,
                ).to_dict()
            )
    return pd.DataFrame(rows)


def make_bsx_grid_features(state: AppState, swing: bool) -> pd.DataFrame:
    selected_pitch_type = state.pitch_type_name
    return make_bsx_grid_features_for_pitch_type(state, swing, selected_pitch_type)


def make_bsx_grid_features_for_pitch_type(
    state: AppState, swing: bool, pitch_type_name: str
) -> pd.DataFrame:
    rows: list[dict] = []
    for x in xs:
        for z in zs:
            rows.append(
                BSXFeatures(
                    p_throws=state.p_throws,
                    stand=state.stand,
                    pitch_type=PITCH_TYPES[pitch_type_name],
                    release_speed=state.release_speed,
                    pfx_x=state.pfx_x,
                    pfx_z=state.pfx_z,
                    px=float(x),
                    pz=float(z),
                    swing=swing,
                ).to_dict()
            )
    return pd.DataFrame(rows)


def snap_to_grid(x: float, y: float) -> tuple[float, float]:
    snapped_x = float(xs[np.abs(xs - x).argmin()])
    y_norm = y / PLATE_HEIGHT
    snapped_z_norm = float(zs[np.abs(zs - y_norm).argmin()])
    return snapped_x, snapped_z_norm


def make_strike_zone_figure(
    pitch_x: float | None,
    pitch_z: float | None,
    outcome_probs: pd.DataFrame | None,
    selected_metric: str,
) -> go.Figure:
    fig = go.Figure()

    gx, gz_norm = np.meshgrid(xs, zs)
    gy = gz_norm * PLATE_HEIGHT
    metric_col = METRIC_COLUMNS[selected_metric]
    metric_label = METRIC_LABELS[selected_metric]

    if outcome_probs is None:
        heatmap_probs = np.full((len(zs), len(xs)), 0.0)
    else:
        grid_keys = pd.MultiIndex.from_product(
            [np.round(xs, 6), np.round(zs, 6)],
            names=["x_key", "z_key"],
        )
        metric_values = outcome_probs.reindex(grid_keys)[metric_col].fillna(0.0).to_numpy()
        heatmap_probs = metric_values.reshape(len(xs), len(zs)).T

    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=gy[:, 0],
            z=heatmap_probs,
            customdata=gz_norm,
            zmin=0.0,
            zmax=1.0,
            colorscale=[
                [0.0, "#2166ac"],
                [1.0, "#ffffff"],
            ],
            showscale=True,
            colorbar=go.heatmap.ColorBar(title=metric_label),
            hovertemplate=f"x=%{{x:.2f}}<br>z=%{{customdata:.2f}}<br>{metric_label}=%{{z:.3f}}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[-1, -1, 1, 1, -1],
            y=[0, PLATE_HEIGHT, PLATE_HEIGHT, 0, 0],
            fill="toself",
            fillcolor="rgba(255,255,255,0.08)",
            marker=go.scatter.Marker(opacity=0),
            line=go.scatter.Line(color="rgba(230,230,230,0.8)"),
            hoverinfo="skip",
        )
    )

    # invisible points to click on
    fig.add_trace(
        go.Scatter(
            x=gx.flatten(),
            y=gy.flatten(),
            mode="markers",
            customdata=np.column_stack((gz_norm.flatten(), heatmap_probs.flatten())),
            marker=go.scatter.Marker(
                color="rgba(0,0,0,0)",
                size=14,
            ),
            hovertemplate=f"x=%{{x:.2f}}<br>z=%{{customdata[0]:.2f}}<br>{metric_label}=%{{customdata[1]:.3f}}<extra></extra>",
        )
    )

    if pitch_x is not None and pitch_z is not None:
        fig.add_trace(
            go.Scatter(
                x=[pitch_x],
                y=[pitch_z * PLATE_HEIGHT],
                mode="markers",
                marker=go.scatter.Marker(color="red", opacity=1, size=10),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        clickmode="event",
        dragmode=False,
        width=1100,
        height=800,
        yaxis=go.layout.YAxis(
            scaleanchor="x",
            scaleratio=1,
            range=list(Y_AXIS_RANGE),
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        xaxis=go.layout.XAxis(
            range=list(X_AXIS_RANGE),
            showticklabels=False,
            fixedrange=True,
        ),
        margin=go.layout.Margin(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def make_plotly_options(fig: go.Figure) -> dict:
    options = fig.to_plotly_json()
    options["config"] = {
        "displayModeBar": False,
        "displaylogo": False,
    }
    return options


PROBABILITY_METRIC_KEYS = ("S", "B", "X", "swing", "take")


def build_app() -> None:
    state = AppState()

    def recompute_and_refresh() -> None:
        precompute_outcome_probs()
        update_chart()
        update_selected_metric_label()

    def precompute_outcome_probs() -> None:
        state.outcome_probs = compute_outcome_probs_for_pitch_type(state.pitch_type_name)

    def compute_outcome_probs_for_pitch_type(pitch_type_name: str) -> pd.DataFrame:
        swing_features_df = make_swing_grid_features(state, pitch_type_name=pitch_type_name)
        p_swing_false, p_swing_true = predict_swing_batch(swing_features_df)
        bsx_features_df_swing = make_bsx_grid_features_for_pitch_type(state, swing=True, pitch_type_name=pitch_type_name)
        bsx_features_df_take = make_bsx_grid_features_for_pitch_type(state, swing=False, pitch_type_name=pitch_type_name)
        bsx_probs_swing = predict_bsx_batch(bsx_features_df_swing)
        bsx_probs_take = predict_bsx_batch(bsx_features_df_take)
        p_strike = bsx_probs_swing["p_S"] * p_swing_true + bsx_probs_take["p_S"] * p_swing_false
        p_ball = bsx_probs_swing["p_B"] * p_swing_true + bsx_probs_take["p_B"] * p_swing_false
        p_in_play = bsx_probs_swing["p_X"] * p_swing_true + bsx_probs_take["p_X"] * p_swing_false
        lookup = pd.DataFrame(
            {
                "x": swing_features_df["px"].astype(float),
                "z": swing_features_df["pz"].astype(float),
                "p_S": p_strike.to_numpy(),
                "p_B": p_ball.to_numpy(),
                "p_X": p_in_play.to_numpy(),
                "p_swing": p_swing_true.to_numpy(),
                "p_take": p_swing_false.to_numpy(),
            }
        )
        lookup["x_key"] = lookup["x"].round(6)
        lookup["z_key"] = lookup["z"].round(6)
        return lookup.set_index(["x_key", "z_key"])

    def find_optimal_pitch_type_and_location() -> tuple[str, float, float, float]:
        metric_col = METRIC_COLUMNS[state.selected_metric]
        find_max = state.optimization_direction == "max"

        best_pitch_type = state.pitch_type_name
        best_x = float(xs[0])
        best_z = float(zs[0])
        best_value = -np.inf if find_max else np.inf

        for pitch_type_name in PITCH_TYPES:
            outcome_probs = compute_outcome_probs_for_pitch_type(pitch_type_name)
            metric_series = outcome_probs[metric_col]
            best_idx = metric_series.idxmax() if find_max else metric_series.idxmin()
            metric_value = float(metric_series.loc[best_idx])
            x_key, z_key = best_idx
            if (find_max and metric_value > best_value) or (not find_max and metric_value < best_value):
                best_pitch_type = pitch_type_name
                best_x = float(x_key)
                best_z = float(z_key)
                best_value = metric_value

        return best_pitch_type, best_x, best_z, best_value

    def position_text() -> str:
        if state.pitch_x is None or state.pitch_z is None:
            return "Click the zone to choose pitch location."
        return f"Selected location: x={state.pitch_x:.2f}, z={state.pitch_z:.2f}"

    def all_metrics_text() -> str | None:
        if state.pitch_x is None or state.pitch_z is None:
            return None
        if state.outcome_probs is None:
            return None

        key = (round(float(state.pitch_x), 6), round(float(state.pitch_z), 6))
        if key not in state.outcome_probs.index:
            return None

        row = state.outcome_probs.loc[key]
        lines: list[str] = []
        for metric_key in PROBABILITY_METRIC_KEYS:
            metric_col = METRIC_COLUMNS[metric_key]
            metric_label = METRIC_LABELS[metric_key]
            metric_value = float(row[metric_col])
            lines.append(f"{metric_label}={metric_value:.3f}")

        return "\n".join(lines)

    def update_chart() -> None:
        chart.figure = make_plotly_options(
            make_strike_zone_figure(
                state.pitch_x,
                state.pitch_z,
                state.outcome_probs,
                state.selected_metric,
            )
        )
        chart.update()

    def update_selected_metric_label() -> None:
        metrics_label.set_text(all_metrics_text() or "")

    precompute_outcome_probs()

    with ui.row().classes("w-full items-start no-wrap gap-4"):
        with ui.column().classes("shrink-0").style(
            "width: 360px; max-height: calc(100vh - 24px); overflow-y: auto; padding-right: 0.5rem;"
        ):
            ui.label("Pitch Explorer").classes("text-h6")

            def on_pitch_type_change(event) -> None:
                state.pitch_type_name = str(event.value)
                recompute_and_refresh()

            def on_release_speed_change(event) -> None:
                state.release_speed = float(event.value)
                speed_label.set_text(f"Release speed: {state.release_speed:.0f} mph")
                recompute_and_refresh()

            def on_metric_change(event) -> None:
                state.selected_metric = str(event.value)
                update_chart()
                update_selected_metric_label()

            def on_find_optimal_click() -> None:
                best_pitch_type, best_x, best_z, best_value = find_optimal_pitch_type_and_location()
                state.pitch_type_name = best_pitch_type
                state.pitch_x = best_x
                state.pitch_z = best_z
                precompute_outcome_probs()
                update_chart()
                selected_location.set_text(position_text())
                update_selected_metric_label()
                if pitch_type_select.value != best_pitch_type:
                    pitch_type_select.set_value(best_pitch_type)
                direction_text = "Maximized" if state.optimization_direction == "max" else "Minimized"
                optimization_result_label.set_text(
                    f"{direction_text} {METRIC_LABELS[state.selected_metric]} with\n"
                    f"Pitch type: {best_pitch_type}\n"
                    f"Location: x={best_x:.2f}, z={best_z:.2f}\n"
                    f"Probability: {best_value:.3f}"
                )

            def previous_pitch_text() -> str:
                prev_pitch_label = state.prev_pitch_type_name or "None"
                prev_type_label = state.prev_type or "None"
                if state.prev_pitch_x is None or state.prev_pitch_z is None:
                    prev_loc_label = "None"
                else:
                    prev_loc_label = f"x={state.prev_pitch_x:.2f}, z={state.prev_pitch_z:.2f}"
                return f"Previous pitch: {prev_pitch_label} | Result: {prev_type_label} | Location: {prev_loc_label}"

            def refresh_previous_pitch_label() -> None:
                previous_pitch_label.set_text(previous_pitch_text())

            def on_use_current_as_previous_click() -> None:
                state.prev_pitch_type_name = state.pitch_type_name
                if state.pitch_x is not None and state.pitch_z is not None:
                    state.prev_pitch_x = float(state.pitch_x)
                    state.prev_pitch_z = float(state.pitch_z)
                if prev_pitch_type_select.value != state.pitch_type_name:
                    prev_pitch_type_select.set_value(state.pitch_type_name)
                if state.prev_pitch_x is not None:
                    prev_pitch_x_number.set_value(state.prev_pitch_x)
                if state.prev_pitch_z is not None:
                    prev_pitch_z_number.set_value(state.prev_pitch_z)
                refresh_previous_pitch_label()
                recompute_and_refresh()

            ui.select(
                options=METRIC_OPTIONS,
                value=state.selected_metric,
                label="Displayed metric",
                on_change=on_metric_change,
            ).classes("w-full")

            selected_location = ui.label(position_text())
            metrics_label = ui.label(all_metrics_text() or "").classes("whitespace-pre-line")
            previous_pitch_label = ui.label(previous_pitch_text()).classes("text-caption")

            with ui.expansion("Game context", icon="sports_baseball", value=True).classes("w-full"):
                with ui.column().classes("w-full gap-3"):
                    with ui.row().classes("w-full gap-3"):
                        ui.select(
                            options=HANDEDNESS_OPTIONS,
                            value=state.p_throws,
                            label="Pitcher hand",
                            on_change=lambda event: (
                                setattr(state, "p_throws", str(event.value)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        ui.select(
                            options=HANDEDNESS_OPTIONS,
                            value=state.stand,
                            label="Batter stand",
                            on_change=lambda event: (
                                setattr(state, "stand", str(event.value)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        ui.select(
                            options=INNING_HALVES,
                            value=state.inning_topbot,
                            label="Inning half",
                            on_change=lambda event: (
                                setattr(state, "inning_topbot", str(event.value)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")

                    with ui.row().classes("w-full gap-3"):
                        ui.number(
                            label="Inning",
                            value=state.inning,
                            min=1,
                            max=15,
                            precision=0,
                            on_change=lambda event: (
                                setattr(state, "inning", int(event.value or 1)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        ui.number(
                            label="Balls",
                            value=state.balls,
                            min=0,
                            max=3,
                            precision=0,
                            on_change=lambda event: (
                                setattr(state, "balls", int(event.value or 0)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        ui.number(
                            label="Strikes",
                            value=state.strikes,
                            min=0,
                            max=2,
                            precision=0,
                            on_change=lambda event: (
                                setattr(state, "strikes", int(event.value or 0)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        ui.number(
                            label="Outs",
                            value=state.outs_when_up,
                            min=0,
                            max=2,
                            precision=0,
                            on_change=lambda event: (
                                setattr(state, "outs_when_up", int(event.value or 0)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")

                    with ui.row().classes("w-full gap-3"):
                        ui.number(
                            label="Batting score",
                            value=state.bat_score,
                            min=0,
                            max=30,
                            precision=0,
                            on_change=lambda event: (
                                setattr(state, "bat_score", int(event.value or 0)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        ui.number(
                            label="Fielding score",
                            value=state.fld_score,
                            min=0,
                            max=30,
                            precision=0,
                            on_change=lambda event: (
                                setattr(state, "fld_score", int(event.value or 0)),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")

                    with ui.row().classes("w-full gap-5"):
                        ui.checkbox(
                            "Runner on 1B",
                            value=state.on_1b,
                            on_change=lambda event: (
                                setattr(state, "on_1b", bool(event.value)),
                                recompute_and_refresh(),
                            ),
                        )
                        ui.checkbox(
                            "Runner on 2B",
                            value=state.on_2b,
                            on_change=lambda event: (
                                setattr(state, "on_2b", bool(event.value)),
                                recompute_and_refresh(),
                            ),
                        )
                        ui.checkbox(
                            "Runner on 3B",
                            value=state.on_3b,
                            on_change=lambda event: (
                                setattr(state, "on_3b", bool(event.value)),
                                recompute_and_refresh(),
                            ),
                        )

                    with ui.row().classes("w-full items-center justify-between gap-1"):
                        ui.slider(
                            min=0.0,
                            max=0.5,
                            step=0.01,
                            value=state.k_rate,
                            on_change=lambda event: (
                                setattr(state, "k_rate", float(event.value)),
                                k_rate_label.set_text(f"Batter K%: {state.k_rate:.2%}"),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        k_rate_label = ui.label(f"Batter K%: {state.k_rate:.2%}").classes("text-caption")

                    with ui.row().classes("w-full items-center justify-between gap-1"):
                        ui.slider(
                            min=0.0,
                            max=0.5,
                            step=0.01,
                            value=state.bb_rate,
                            on_change=lambda event: (
                                setattr(state, "bb_rate", float(event.value)),
                                bb_rate_label.set_text(f"Batter BB%: {state.bb_rate:.2%}"),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        bb_rate_label = ui.label(f"Batter BB%: {state.bb_rate:.2%}").classes("text-caption")

                    with ui.separator().classes("w-full"):
                        pass

                    ui.label("Previous pitch context").classes("text-subtitle2")

                    prev_pitch_type_options = {"none": "None", **{name: name for name in PITCH_TYPES}}
                    prev_pitch_type_select = ui.select(
                        options=prev_pitch_type_options,
                        value="none" if state.prev_pitch_type_name is None else state.prev_pitch_type_name,
                        label="Previous pitch type",
                        on_change=lambda event: (
                            setattr(
                                state,
                                "prev_pitch_type_name",
                                None if str(event.value) == "none" else str(event.value),
                            ),
                            refresh_previous_pitch_label(),
                            recompute_and_refresh(),
                        ),
                    ).classes("w-full")

                    ui.select(
                        options=PREV_PITCH_RESULT_OPTIONS,
                        value="none" if state.prev_type is None else state.prev_type,
                        label="Previous pitch result",
                        on_change=lambda event: (
                            setattr(state, "prev_type", None if str(event.value) == "none" else str(event.value)),
                            refresh_previous_pitch_label(),
                            recompute_and_refresh(),
                        ),
                    ).classes("w-full")

                    with ui.row().classes("w-full gap-3"):
                        prev_pitch_x_number = ui.number(
                            label="Previous pitch x",
                            value=state.prev_pitch_x,
                            min=float(X_AXIS_RANGE[0]),
                            max=float(X_AXIS_RANGE[1]),
                            step=0.01,
                            on_change=lambda event: (
                                setattr(
                                    state,
                                    "prev_pitch_x",
                                    None if event.value in (None, "") else float(event.value),
                                ),
                                refresh_previous_pitch_label(),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        prev_pitch_z_number = ui.number(
                            label="Previous pitch z",
                            value=state.prev_pitch_z,
                            min=float(zs.min()),
                            max=float(zs.max()),
                            step=0.01,
                            on_change=lambda event: (
                                setattr(
                                    state,
                                    "prev_pitch_z",
                                    None if event.value in (None, "") else float(event.value),
                                ),
                                refresh_previous_pitch_label(),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")

                    ui.button("Use current pitch as previous", on_click=on_use_current_as_previous_click).classes("w-full")

            with ui.expansion("Pitch controls", icon="tune", value=False).classes("w-full"):
                with ui.column().classes("w-full gap-3"):
                    pitch_type_select = ui.select(
                        options=list(PITCH_TYPES.keys()),
                        value=state.pitch_type_name,
                        label="Pitch type",
                        on_change=on_pitch_type_change,
                    ).classes("w-full")

                    with ui.row().classes("w-full items-center justify-between gap-1"):
                        ui.slider(
                            min=70,
                            max=105,
                            step=1,
                            value=state.release_speed,
                            on_change=on_release_speed_change,
                        ).classes("w-full")
                        speed_label = ui.label(f"Release speed: {state.release_speed:.0f} mph").classes("text-caption")

                    with ui.row().classes("w-full items-center justify-between gap-1"):
                        ui.slider(
                            min=-2,
                            max=2,
                            step=0.05,
                            value=state.pfx_x,
                            on_change=lambda event: (
                                setattr(state, "pfx_x", float(event.value)),
                                pfx_x_label.set_text(f"pfx_x: {state.pfx_x:.2f}"),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        pfx_x_label = ui.label(f"pfx_x: {state.pfx_x:.2f}").classes("text-caption")

                    with ui.row().classes("w-full items-center justify-between gap-1"):
                        ui.slider(
                            min=-2,
                            max=2,
                            step=0.05,
                            value=state.pfx_z,
                            on_change=lambda event: (
                                setattr(state, "pfx_z", float(event.value)),
                                pfx_z_label.set_text(f"pfx_z: {state.pfx_z:.2f}"),
                                recompute_and_refresh(),
                            ),
                        ).classes("w-full")
                        pfx_z_label = ui.label(f"pfx_z: {state.pfx_z:.2f}").classes("text-caption")

            with ui.expansion("Optimization", icon="query_stats", value=False).classes("w-full"):
                with ui.column().classes("w-full gap-3"):
                    ui.select(
                        options=OPTIMIZATION_DIRECTIONS,
                        value=state.optimization_direction,
                        label="Objective",
                        on_change=lambda event: setattr(state, "optimization_direction", str(event.value)),
                    ).classes("w-full")
                    ui.button("Find best pitch type + location", on_click=on_find_optimal_click).classes("w-full")
                    optimization_result_label = ui.label("").classes("whitespace-pre-line text-caption")

        with ui.column().classes("flex-1 min-w-0 items-center"):
            chart = ui.plotly(
                make_plotly_options(
                    make_strike_zone_figure(
                        state.pitch_x,
                        state.pitch_z,
                        state.outcome_probs,
                        state.selected_metric,
                    )
                )
            ).classes("w-full").style("max-width: 1100px;")

            def on_plot_click(event) -> None:
                args = event.args or {}
                points = args.get("points") or []
                if not points:
                    return
                point = points[0]
                clicked_x = float(point["x"])
                clicked_y = float(point["y"])
                state.pitch_x, state.pitch_z = snap_to_grid(clicked_x, clicked_y)
                update_chart()
                selected_location.set_text(position_text())
                update_selected_metric_label()

            chart.on("plotly_click", on_plot_click)

    precompute_outcome_probs()


build_app()
ui.run(title="Pitch Explorer")
