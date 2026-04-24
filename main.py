import streamlit as st

from backend.navigation import render_current_page


def main() -> None:
    st.set_page_config(page_title="Display Project", layout="wide")
    render_current_page()


if __name__ == "__main__":
    main()
