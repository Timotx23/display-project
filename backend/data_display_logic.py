import streamlit as st
import numpy as np
from typing import Iterator

from backend.state import (
    LANDING_PAGE,
    DisplayState,
    get_display_state as get_saved_display_state,
    set_current_page,
    set_episode,
)
from backend.video import MultiCameraFrame, VideoService


def reset_to_landing() -> None:
    set_current_page(LANDING_PAGE)
    st.rerun()


def get_display_state() -> DisplayState:
    return get_saved_display_state()


def load_frames_for_current_selection() -> list[np.ndarray]:
    display_state = get_saved_display_state()
    service = VideoService()
    return service.load_episode_frames(
        database_path=display_state.database_path,
        episode=display_state.episode,
    )


def iter_frames_for_current_selection() -> Iterator[np.ndarray]:
    display_state = get_saved_display_state()
    service = VideoService()
    return service.iter_episode_frames(
        database_path=display_state.database_path,
        episode=display_state.episode,
    )


def iter_multicamera_frames_for_current_selection() -> Iterator[MultiCameraFrame]:
    display_state = get_saved_display_state()
    service = VideoService()
    return service.iter_multicamera_episode_frames(
        database_path=display_state.database_path,
        episode=display_state.episode,
    )


def get_prompt_for_current_selection() -> str | None:
    display_state = get_saved_display_state()
    service = VideoService()
    return service.get_episode_prompt(
        database_path=display_state.database_path,
        episode=display_state.episode,
    )


def set_episode_for_current_selection(episode: int) -> None:
    set_episode(max(0, int(episode)))
