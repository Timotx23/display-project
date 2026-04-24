from data_display_page import render_data_display_page
from landing_page import render_landing_page

from backend.state import DATA_DISPLAY_PAGE, get_current_page, initialize_state


def render_current_page() -> None:
    initialize_state()

    current_page = get_current_page()

    if current_page == DATA_DISPLAY_PAGE:
        render_data_display_page()
    else:
        render_landing_page()
