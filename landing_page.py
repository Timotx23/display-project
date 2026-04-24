import streamlit as st

from backend.landing_logic import handle_start_click


def render_landing_page() -> None:
    st.title("Landing Page")
    st.write("Enter your configuration to start the data display.")

    database_path = st.text_input("Database path *")
    fps = 10
    episode = st.number_input("Episode", min_value=0, value=0, step=1)

    if st.button("Start"):
        handle_start_click(
            database_path=database_path,
            fps=int(fps),
            episode=int(episode),
        )
