# app.py
# ------------------------------------------------------------------
# Project: AI-Assisted OpenMP Scheduling for Parallel Numerical Computation
# Description: Academic Streamlit UI for OpenMP Pi computations,
#              parallel metrics evaluation, live schedule comparisons, and AI predictions.
# ------------------------------------------------------------------

import os
import streamlit as st

# Configure Page Layout & Icon
st.set_page_config(
    page_title="Parallel Performance Lab — AI-Assisted OpenMP Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Theme & CSS Styles
from ui.styles import apply_custom_styles
apply_custom_styles()

# Render Navigation Sidebar
from ui.sidebar import render_sidebar
selected_page = render_sidebar()

# Import Page Modules
from ui.dashboard import render_dashboard
from ui.benchmark import render_benchmark_lab
from ui.scheduling import render_scheduling_analysis
from ui.history import render_experiment_history
from ui.ai_optimizer import render_ai_optimizer
from ui.model_insights import render_model_insights
from ui.system_info import render_system_info

# Page Routing Logic
if selected_page == "Dashboard":
    render_dashboard()
elif selected_page == "Benchmark Lab":
    render_benchmark_lab()
elif selected_page == "Scheduling Analysis":
    render_scheduling_analysis()
elif selected_page == "Experiment History":
    render_experiment_history()
elif selected_page == "AI Optimizer":
    render_ai_optimizer()
elif selected_page == "Model Insights":
    render_model_insights()
elif selected_page == "System Information":
    render_system_info()
else:
    render_dashboard()
