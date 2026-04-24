from dataclasses import dataclass

import streamlit as st

CURRENT_PAGE_KEY = "current_page"
DATABASE_PATH_KEY = "database_path"
FPS_KEY = "fps"
EPISODE_KEY = "episode"

LANDING_PAGE = "landing_page"
DATA_DISPLAY_PAGE = "data_display_page"


@dataclass(frozen=True)
class DisplayState:
    database_path: str
    fps: int
    episode: int


def initialize_state() -> None:
    st.session_state.setdefault(CURRENT_PAGE_KEY, LANDING_PAGE)
    st.session_state.setdefault(DATABASE_PATH_KEY, "")
    st.session_state.setdefault(FPS_KEY, 10)
    st.session_state.setdefault(EPISODE_KEY, 0)


def set_current_page(page_name: str) -> None:
    st.session_state[CURRENT_PAGE_KEY] = page_name


def get_current_page() -> str:
    return st.session_state[CURRENT_PAGE_KEY]


def save_display_state(database_path: str, fps: int, episode: int) -> None:
    st.session_state[DATABASE_PATH_KEY] = database_path
    st.session_state[FPS_KEY] = fps
    st.session_state[EPISODE_KEY] = episode


def set_episode(episode: int) -> None:
    st.session_state[EPISODE_KEY] = episode


def get_display_state() -> DisplayState:
    return DisplayState(
        database_path=st.session_state[DATABASE_PATH_KEY],
        fps=st.session_state[FPS_KEY],
        episode=st.session_state[EPISODE_KEY],
    )
