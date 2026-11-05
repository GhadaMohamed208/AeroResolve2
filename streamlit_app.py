"""Entry point for Streamlit (local run and Streamlit Community Cloud): streamlit run streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ui.streamlit_app import main  # noqa: E402

main()
