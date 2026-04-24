import streamlit as st

from backend.state import DATA_DISPLAY_PAGE, save_display_state, set_current_page


def handle_start_click(database_path: str, fps: int, episode: int) -> None:
    if not database_path.strip():
        st.error("Database path is required.")
        return

    save_display_state(
        database_path=database_path.strip(),
        fps=fps,
        episode=episode,
    )
    set_current_page(DATA_DISPLAY_PAGE)
    st.rerun()
