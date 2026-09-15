# ui/sidebar.py
import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 4px 18px 4px; border-bottom: 1px solid #1E293B; margin-bottom: 16px;">
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

        cur = st.session_state["selected_page"]

        # OVERVIEW
        st.markdown("""<div style="font-size: 10px; font-weight: 800; color: #64748B; letter-spacing: 1.2px; margin-bottom: 6px;">OVERVIEW</div>""", unsafe_allow_html=True)
        if st.button("📊  Dashboard",
                     use_container_width=True,
                     type="primary" if cur == "Dashboard" else "secondary"):
            set_page("Dashboard"); st.rerun()

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # EXPERIMENTAL BENCHMARKS
        st.markdown("""<div style="font-size: 10px; font-weight: 800; color: #64748B; letter-spacing: 1.2px; margin-bottom: 6px;">EXPERIMENTS</div>""", unsafe_allow_html=True)
        if st.button("⚡  Scheduling Analysis",
                     use_container_width=True,
                     type="primary" if cur == "Scheduling Analysis" else "secondary"):
            set_page("Scheduling Analysis"); st.rerun()

        if st.button("🔬  Benchmark Lab",
                     use_container_width=True,
                     type="primary" if cur == "Benchmark Lab" else "secondary"):
            set_page("Benchmark Lab"); st.rerun()

        if st.button("📜  Experiment History",
                     use_container_width=True,
                     type="primary" if cur == "Experiment History" else "secondary"):
            set_page("Experiment History"); st.rerun()

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # MACHINE LEARNING
        st.markdown("""<div style="font-size: 10px; font-weight: 800; color: #64748B; letter-spacing: 1.2px; margin-bottom: 6px;">INTELLIGENCE</div>""", unsafe_allow_html=True)
        if st.button("🤖  AI Optimizer",
                     use_container_width=True,
                     type="primary" if cur == "AI Optimizer" else "secondary"):
            set_page("AI Optimizer"); st.rerun()

        if st.button("🧠  Model Insights",
                     use_container_width=True,
                     type="primary" if cur == "Model Insights" else "secondary"):
            set_page("Model Insights"); st.rerun()

        # Engine Status
        st.markdown("""
        <div style="margin-top: 24px; padding: 12px; background: #131B2E;
                    border: 1px solid #1E293B; border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="status-dot"></span>
                <span style="font-size: 11px; font-weight: 700; color: #10B981;">Engine Ready</span>
            </div>
            <div style="font-size: 11px; color: #64748B; margin-top: 4px;">
                OpenMP C Binaries + ML Ready
            </div>
        </div>
        """, unsafe_allow_html=True)

    return st.session_state["selected_page"]
