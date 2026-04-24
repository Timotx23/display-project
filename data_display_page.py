import streamlit as st
from time import sleep
import pandas as pd
import altair as alt
import numpy as np

from backend.data_display_logic import (
    get_display_state,
    get_prompt_for_current_selection,
    iter_multicamera_frames_for_current_selection,
    reset_to_landing,
    set_episode_for_current_selection,
)


def _sanitize_joint_angles(raw_angles: object) -> list[float] | None:
    """Return exactly 6 numeric joint angles, or None if invalid."""
    if not isinstance(raw_angles, (list, tuple)):
        return None
    if len(raw_angles) < 6:
        return None

    sanitized: list[float] = []
    for value in raw_angles[:6]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        sanitized.append(numeric)
    return sanitized


def _get_recent_steps(history: list[list[float]], window_size: int = 5) -> list[list[float]]:
    """Return the latest up-to-window_size valid joint-angle steps."""
    if window_size <= 0:
        return history
    return history[-window_size:]


def _calculate_moving_average_joint_angles(recent_steps: list[list[float]]) -> list[float]:
    """Average each joint angle over the provided recent steps."""
    if not recent_steps:
        return []
    return np.mean(np.array(recent_steps, dtype=float), axis=0).tolist()


def _calculate_moving_average_delta_joint_angles(recent_steps: list[list[float]]) -> list[float]:
    """
    Compute per-joint average delta from consecutive steps in the recent window.
    If there is only one step, return zeros.
    """
    if not recent_steps:
        return []
    if len(recent_steps) == 1:
        return [0.0] * 6

    step_array = np.array(recent_steps, dtype=float)
    deltas = np.diff(step_array, axis=0)
    return np.mean(deltas, axis=0).tolist()


def render_data_display_page() -> None:
    st.title("Data Display")

    display_state = get_display_state()
    st.write(f"Database path: `{display_state.database_path}`")
    st.write(f"FPS: `{display_state.fps}`")
    selected_episode = st.number_input(
        "Episode",
        min_value=0,
        value=int(display_state.episode),
        step=1,
    )
    if int(selected_episode) != int(display_state.episode):
        set_episode_for_current_selection(int(selected_episode))
        st.rerun()

    left_column, right_column = st.columns([2, 2])
    prompt_text = None
    try:
        prompt_text = get_prompt_for_current_selection()
    except Exception:
        prompt_text = None

    left_column.subheader("Action Prompt")
    if prompt_text:
        left_column.info(prompt_text)
    else:
        left_column.caption("No prompt found in DB for this episode.")

    main_camera_column, wrist_camera_column = left_column.columns(2)
    main_camera_column.subheader("Main Camera")
    main_frame_placeholder = main_camera_column.empty()
    play_video_clicked = left_column.button("Play Video")
    wrist_camera_column.subheader("Wrist Camera")
    wrist_frame_placeholder = wrist_camera_column.empty()
    right_column.subheader("Absolute Joint Angles")
    absolute_chart_placeholder = right_column.empty()
    right_column.subheader("Delta Joint Angles")
    delta_chart_placeholder = right_column.empty()
    right_column.subheader("Sync Monitor")
    frame_counter_placeholder = right_column.empty()
    sync_status_placeholder = right_column.empty()

    if play_video_clicked:
        frame_delay_seconds = 1.0 / max(1, display_state.fps)
        rendered_frames = 0
        wrist_rendered_frames = 0
        joint_chart_updates = 0
        joint_angle_history: list[list[float]] = []
        current_step_index = -1
        try:
            for current_step_index, camera_frame in enumerate(iter_multicamera_frames_for_current_selection()):
                main_frame_placeholder.image(camera_frame.main_frame, channels="RGB", width="content")
                if camera_frame.wrist_frame is not None:
                    wrist_frame_placeholder.image(camera_frame.wrist_frame, channels="RGB", width="content")
                    wrist_rendered_frames += 1
                rendered_frames += 1

                sanitized_joint_angles = _sanitize_joint_angles(camera_frame.joint_angles)
                if sanitized_joint_angles is not None:
                    joint_angle_history.append(sanitized_joint_angles)

                # Update charts on every step from the same step stream as video.
                if joint_angle_history:
                    joint_labels = [f"Joint {joint_idx}" for joint_idx in range(1, 7)]
                    recent_steps = _get_recent_steps(joint_angle_history, window_size=5)
                    absolute_values = _calculate_moving_average_joint_angles(recent_steps)
                    delta_values = _calculate_moving_average_delta_joint_angles(recent_steps)
                    joint_chart_updates += 1

                    absolute_df = pd.DataFrame({"Joint": joint_labels, "Angle": absolute_values})
                    delta_df = pd.DataFrame({"Joint": joint_labels, "Delta": delta_values})

                    absolute_chart = (
                        alt.Chart(absolute_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Joint:N", sort=joint_labels),
                            y=alt.Y("Angle:Q", scale=alt.Scale(zero=True)),
                            tooltip=["Joint", alt.Tooltip("Angle:Q", format=".3f")],
                        )
                        .properties(height=200)
                    )

                    delta_chart = (
                        alt.Chart(delta_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Joint:N", sort=joint_labels),
                            y=alt.Y("Delta:Q", scale=alt.Scale(zero=True)),
                            color=alt.condition(alt.datum.Delta >= 0, alt.value("#2E7D32"), alt.value("#C62828")),
                            tooltip=["Joint", alt.Tooltip("Delta:Q", format=".3f")],
                        )
                        .properties(height=200)
                    )
                    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#555").encode(y="y:Q")

                    absolute_chart_placeholder.altair_chart(absolute_chart, use_container_width=True)
                    delta_chart_placeholder.altair_chart(delta_chart + zero_line, use_container_width=True)

                frame_counter_placeholder.markdown(
                    f"**Step frame index:** `{current_step_index}`\n\n"
                    f"**Main frames shown:** `{rendered_frames}`\n\n"
                    f"**Wrist frames shown:** `{wrist_rendered_frames}`\n\n"
                    f"**Joint chart updates (every 5 steps):** `{joint_chart_updates}`"
                )
                if camera_frame.wrist_frame is not None:
                    sync_status_placeholder.success("In sync on this step (both frames present).")
                else:
                    sync_status_placeholder.warning("Main frame present, wrist frame missing on this step.")
                sleep(frame_delay_seconds)
        except Exception as error:
            st.error(f"Could not load or play video: {error}")
            rendered_frames = 0

        if rendered_frames == 0:
            st.info("No frames available for this episode.")
        else:
            st.success(
                f"Played `{rendered_frames}` main frames and `{wrist_rendered_frames}` wrist frames."
            )
            if rendered_frames == wrist_rendered_frames:
                sync_status_placeholder.success("Playback complete: both feeds stayed fully in sync.")
            else:
                sync_status_placeholder.warning(
                    "Playback complete: some wrist frames were missing, but frame index stayed shared."
                )

    if st.button("Back to Landing Page"):
        reset_to_landing()
