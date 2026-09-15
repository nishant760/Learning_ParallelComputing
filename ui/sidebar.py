# ui/sidebar.py
import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 12px 4px 20px 4px; border-bottom: 1px solid #1E293B; margin-bottom: 20px;">
            <div style="font-size: 17px; font-weight: 800; color: #38BDF8; letter-spacing: 0.5px;">
                ⚡ PARALLEL LAB
            </div>
            <div style="font-size: 12px; font-weight: 600; color: #F8FAFC; margin-top: 3px;">
                AI-Assisted OpenMP Scheduling
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "selected_page" not in st.session_state:
            st.session_state["selected_page"] = "Dashboard"

        def set_page(page_name):
            st.session_state["selected_page"] = page_name

        st.markdown("""<div style="font-size: 10px; font-weight: 800; color: #475569; letter-spacing: 1.2px; margin-bottom: 8px;">OVERVIEW</div>""", unsafe_allow_html=True)
        if st.button("📊  Dashboard", use_container_width=True, type="primary" if st.session_state["selected_page"] == "Dashboard" else "secondary"):
            set_page("Dashboard")
            st.rerun()

        st.markdown("""<div style="font-size: 10px; font-weight: 800; color: #475569; letter-spacing: 1.2px; margin-top: 18px; margin-bottom: 8px;">EXPERIMENTS</div>""", unsafe_allow_html=True)
        if st.button("🔬  Benchmark Lab", use_container_width=True, type="primary" if st.session_state["selected_page"] == "Benchmark Lab" else "secondary"):
            set_page("Benchmark Lab")
            st.rerun()
        if st.button("⚡  Scheduling Analysis", use_container_width=True, type="primary" if st.session_state["selected_page"] == "Scheduling Analysis" else "secondary"):
            set_page("Scheduling Analysis")
            st.rerun()
        if st.button("📜  Experiment History", use_container_width=True, type="primary" if st.session_state["selected_page"] == "Experiment History" else "secondary"):
            set_page("Experiment History")
            st.rerun()

        st.markdown("""<div style="font-size: 10px; font-weight: 800; color: #475569; letter-spacing: 1.2px; margin-top: 18px; margin-bottom: 8px;">INTELLIGENCE</div>""", unsafe_allow_html=True)
        if st.button("🤖  AI Optimizer", use_container_width=True, type="primary" if st.session_state["selected_page"] == "AI Optimizer" else "secondary"):
            set_page("AI Optimizer")
            st.rerun()
        if st.button("🧠  Model Insights", use_container_width=True, type="primary" if st.session_state["selected_page"] == "Model Insights" else "secondary"):
            set_page("Model Insights")
            st.rerun()

        st.markdown("""<div style="font-size: 10px; font-weight: 800; color: #475569; letter-spacing: 1.2px; margin-top: 18px; margin-bottom: 8px;">SYSTEM</div>""", unsafe_allow_html=True)
        if st.button("💻  System Information", use_container_width=True, type="primary" if st.session_state["selected_page"] == "System Information" else "secondary"):
            set_page("System Information")
            st.rerun()


        # Engine Status Box
        st.markdown("""
        <div style="margin-top: 24px; padding: 14px; background: #131B2E; border: 1px solid #1E293B; border-radius: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="status-dot"></span>
                <span style="font-size: 12px; font-weight: 700; color: #10B981;">Engine Ready</span>
            </div>
            <div style="font-size: 11px; color: #64748B; margin-top: 4px;">
                OpenMP Engine Connected
            </div>
        </div>
        """, unsafe_allow_html=True)

    return st.session_state["selected_page"]
