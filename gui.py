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

PLATE_HEIGHT = 3.0
X_AXIS_RANGE = (-1.35, 1.35)
Y_AXIS_RANGE = (-0.35, PLATE_HEIGHT + 0.35)

give = 1
xs = np.linspace(-1 - give, 1 + give, 40)
zs = np.linspace(Y_AXIS_RANGE[0] / PLATE_HEIGHT, Y_AXIS_RANGE[1] / PLATE_HEIGHT, 40)


@dataclass
class AppState:
    release_speed: float = 90
    pitch_type_name: str = next(iter(PITCH_TYPES))
    pitch_x: float | None = None
    pitch_z: float | None = None
    balls: int = 0
    strikes: int = 0
    swing_features: SwingFeatures | None = None
    selected_metric: str = "S"
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


def make_swing_grid_features(state: AppState) -> pd.DataFrame:
    rows: list[dict] = []
    for x in xs:
        for z in zs:
            rows.append(
                SwingFeatures(
                    p_throws="L",
                    stand="R",
                    inning_topbot="Top",
                    prev_pitch_type=None,
                    prev_type=None,
                    pitch_type=PITCH_TYPES[state.pitch_type_name],
                    inning=9,
                    bat_score=0,
                    fld_score=0,
                    outs_when_up=0,
                    on_3b=True,
                    on_2b=True,
                    on_1b=True,
                    balls=state.balls,
                    strikes=state.strikes,
                    release_speed=state.release_speed,
                    pfx_x=0,
                    pfx_z=0,
                    K_rate=0.25,
                    BB_rate=0.25,
                    px=float(x),
                    pz=float(z),
                    prev_px=None,
                    prev_pz=None,
                ).to_dict()
            )
    return pd.DataFrame(rows)


def make_bsx_grid_features(state: AppState, swing: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for x in xs:
        for z in zs:
            rows.append(
                BSXFeatures(
                    p_throws="L",
                    stand="R",
                    pitch_type=PITCH_TYPES[state.pitch_type_name],
                    release_speed=state.release_speed,
                    pfx_x=0,
                    pfx_z=0,
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


PROBABILITY_METRIC_KEYS = ("S", "B", "X", "swing", "take")


def build_app() -> None:
    state = AppState()

    def precompute_outcome_probs() -> None:
        swing_features_df = make_swing_grid_features(state)
        p_swing_false, p_swing_true = predict_swing_batch(swing_features_df)
        bsx_features_df_swing = make_bsx_grid_features(state, swing=True)
        bsx_features_df_take = make_bsx_grid_features(state, swing=False)
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
        state.outcome_probs = lookup.set_index(["x_key", "z_key"])

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
        chart.figure = make_strike_zone_figure(
            state.pitch_x,
            state.pitch_z,
            state.outcome_probs,
            state.selected_metric,
        )
        chart.update()

    def update_selected_metric_label() -> None:
        selected_metric_label.set_text(all_metrics_text() or "")

    precompute_outcome_probs()

    with ui.row().classes("w-full items-start no-wrap"):
        with ui.column().classes("w-90"):
            ui.label("Pitch Explorer").classes("text-h6")

            def on_pitch_type_change(event) -> None:
                state.pitch_type_name = str(event.value)
                precompute_outcome_probs()
                update_chart()
                update_selected_metric_label()

            def on_release_speed_change(event) -> None:
                state.release_speed = float(event.value)
                speed_label.set_text(f"Release speed: {state.release_speed:.0f} mph")
                precompute_outcome_probs()
                update_chart()
                update_selected_metric_label()

            def on_metric_change(event) -> None:
                state.selected_metric = str(event.value)
                update_chart()
                update_selected_metric_label()

            with ui.row().classes("w-full items-center justify-between gap-3"):
                ui.slider(
                    min=70,
                    max=105,
                    step=1,
                    value=state.release_speed,
                    on_change=on_release_speed_change,
                ).classes("w-full")
                speed_label = ui.label(f"Release speed: {state.release_speed:.0f} mph").classes("text-caption")

            ui.select(
                options=list(PITCH_TYPES.keys()),
                value=state.pitch_type_name,
                label="Pitch type",
                on_change=on_pitch_type_change,
            ).classes("w-full")
            ui.select(
                options=METRIC_OPTIONS,
                value=state.selected_metric,
                label="Displayed metric",
                on_change=on_metric_change,
            ).classes("w-full")

            selected_location = ui.label(position_text())

            # count = ui.label(count_text() or "")
            selected_metric_label = ui.label(all_metrics_text() or "").classes("whitespace-pre-line")

        with ui.column().classes("w-full"):
            chart = ui.plotly(
                make_strike_zone_figure(
                    state.pitch_x,
                    state.pitch_z,
                    state.outcome_probs,
                    state.selected_metric,
                )
            ).classes("w-full")

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
