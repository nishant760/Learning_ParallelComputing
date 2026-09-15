# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import streamlit as st

st.set_page_config(
    page_title="Parallel Performance Lab — AI-Assisted OpenMP Scheduling",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ui.styles import apply_custom_styles
apply_custom_styles()

from ui.sidebar import render_sidebar
selected_page = render_sidebar()

from ui.dashboard       import render_dashboard
from ui.scheduling      import render_scheduling_analysis
from ui.benchmark       import render_benchmark_lab
from ui.history         import render_experiment_history
from ui.ai_optimizer    import render_ai_optimizer
from ui.model_insights  import render_model_insights

if selected_page == "Dashboard":
    render_dashboard()
elif selected_page == "Scheduling Analysis":
    render_scheduling_analysis()
elif selected_page == "Benchmark Lab":
    render_benchmark_lab()
elif selected_page == "Experiment History":
    render_experiment_history()
elif selected_page == "AI Optimizer":
    render_ai_optimizer()
elif selected_page == "Model Insights":
    render_model_insights()
else:
    render_dashboard()
