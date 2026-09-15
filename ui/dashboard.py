# ui/dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ui.utils import (
    load_dataset,
    load_model,
    get_dataset_kpis,
    get_ai_recommendation,
)

def render_dashboard():
    # Header Banner
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("""
        <div class="page-header">
            <div class="title">⚡ Parallel Performance Lab</div>
            <div class="subtitle">AI-Assisted Parallel Numerical Computation and Scheduling</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown("""
        <div style="text-align: right; padding-top: 4px;">
            <span class="status-badge"><span class="status-dot"></span> System Ready</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Run Benchmark", use_container_width=True, type="primary"):
            st.session_state["selected_page"] = "Benchmark Lab"
            st.rerun()

    # Problem statement context strip
    st.markdown("""
    <div style="background:#131B2E; border:1px solid #1E293B; border-left:4px solid #38BDF8;
                border-radius:10px; padding:16px 20px; margin-bottom:24px;">
        <div style="font-size:11px; font-weight:700; color:#38BDF8; letter-spacing:1px;
                    text-transform:uppercase; margin-bottom:10px;">Problem Statement</div>
        <div style="font-size:13px; color:#CBD5E1; line-height:1.7;">
            Large-scale numerical computations (<b>π via midpoint rule integration</b>) become bottlenecked on a single core.
            This project parallelises the workload using <b>OpenMP</b> and evaluates three loop scheduling strategies —
            <b style="color:#38BDF8;">Static</b>, <b style="color:#818CF8;">Dynamic</b>, and <b style="color:#10B981;">Guided</b> —
            across varying thread counts and chunk sizes. A <b>Random Forest regression model</b> is trained on empirical
            benchmark data to <b>predict execution time and speedup</b> for unseen workloads, enabling AI-assisted
            scheduling decisions before execution.
        </div>
        <div style="display:flex; gap:24px; margin-top:14px; flex-wrap:wrap;">
            <div style="font-size:12px; color:#64748B;">
                <span style="color:#10B981; font-weight:700;">S = T₁ / Tₚ</span>&nbsp; Speedup
            </div>
            <div style="font-size:12px; color:#64748B;">
                <span style="color:#38BDF8; font-weight:700;">E = S / P × 100%</span>&nbsp; Efficiency
            </div>
            <div style="font-size:12px; color:#64748B;">
                <span style="color:#818CF8; font-weight:700;">1 warmup + 5 measured runs</span>&nbsp; Median timing
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 6 KPI Cards Row
    kpis = get_dataset_kpis()
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">LAST WORKLOAD</div>
        <div class="kpi-value" style="font-size: 18px;">{kpis['last_workload']}</div>
        <div class="kpi-sub">Empirical N</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">BEST SPEEDUP</div>
        <div class="kpi-value">{kpis['best_speedup']}</div>
        <div class="kpi-sub">Peak Measured</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">BEST THREADS</div>
        <div class="kpi-value" style="font-size: 18px;">{kpis['best_threads']}</div>
        <div class="kpi-sub">Optimal Cores</div>
    </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">BEST SCHEDULE</div>
        <div class="kpi-value" style="font-size: 18px;">{kpis['best_schedule']}</div>
        <div class="kpi-sub">Policy Strategy</div>
    </div>
    """, unsafe_allow_html=True)

    c5.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">MODEL R²</div>
        <div class="kpi-value">{kpis['model_r2']}</div>
        <div class="kpi-sub">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

    c6.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">EXPERIMENTS</div>
        <div class="kpi-value">{kpis['experiments_count']}</div>
        <div class="kpi-sub">Dataset Runs</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    df = load_dataset()
    model = load_model()

    # Performance Charts Row
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("""<div class="card-box-header">📈 Performance Overview: Workload vs Speedup</div>""", unsafe_allow_html=True)
        if df is not None and not df.empty:
            std_df = df[(df['schedule'] == 'static') & (df['chunk'] == 1000)].copy()
            fig, ax = plt.subplots(figsize=(6, 3.6))
            fig.patch.set_facecolor('#131B2E')
            ax.set_facecolor('#090D16')
            
            threads_list = [1, 2, 4, 8]
            colors = ['#38BDF8', '#818CF8', '#34D399', '#F43F5E']
            
            for idx, t in enumerate(threads_list):
                sub = std_df[std_df['threads'] == t].sort_values('N')
                if not sub.empty:
                    x_vals = sub['N'] / 1e6
                    ax.plot(x_vals, sub['speedup'], marker='o', markersize=5, linewidth=2.2, color=colors[idx], label=f"{t} Thread{'s' if t>1 else ''}")

            ax.set_title("Workload Size N (Millions) vs Speedup", color='#F8FAFC', fontsize=11, fontweight='bold', pad=12)
            ax.set_xlabel("Workload Size N (Millions)", color='#94A3B8', fontsize=9)
            ax.set_ylabel("Measured Speedup", color='#94A3B8', fontsize=9)
            ax.tick_params(colors='#94A3B8', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.15, color='#94A3B8')
            ax.legend(facecolor='#131B2E', edgecolor='#1E293B', labelcolor='#F8FAFC', fontsize=8)
            for spine in ax.spines.values():
                spine.set_color('#1E293B')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No benchmark dataset available. Run benchmark to generate charts.")

    with col_g2:
        st.markdown("""<div class="card-box-header">⚡ Scheduling Strategy Comparison</div>""", unsafe_allow_html=True)
        if df is not None and not df.empty:
            large_N = df['N'].max()
            sched_df = df[(df['N'] == large_N) & (df['threads'] == 8)].copy()
            if not sched_df.empty:
                fig, ax = plt.subplots(figsize=(6, 3.6))
                fig.patch.set_facecolor('#131B2E')
                ax.set_facecolor('#090D16')

                pivot_df = sched_df.pivot(index='chunk', columns='schedule', values='speedup')
                available_cols = [c for c in ['static', 'dynamic', 'guided'] if c in pivot_df.columns]
                pivot_df = pivot_df[available_cols]

                pivot_df.plot(kind='bar', ax=ax, width=0.65, color=['#38BDF8', '#818CF8', '#34D399'])
                ax.set_title(f"Static vs Dynamic vs Guided (N = {large_N//1000000}M, 8 Threads)", color='#F8FAFC', fontsize=11, fontweight='bold', pad=12)
                ax.set_xlabel("Chunk Size", color='#94A3B8', fontsize=9)
                ax.set_ylabel("Speedup", color='#94A3B8', fontsize=9)
                ax.tick_params(colors='#94A3B8', labelsize=8)
                ax.grid(axis='y', linestyle='--', alpha=0.15, color='#94A3B8')
                ax.legend(facecolor='#131B2E', edgecolor='#1E293B', labelcolor='#F8FAFC', fontsize=8)
                plt.xticks(rotation=0)
                for spine in ax.spines.values():
                    spine.set_color('#1E293B')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No scheduling data found for highest N.")
        else:
            st.info("No benchmark dataset available. Run benchmark to generate charts.")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Execution Table & AI Recommendation
    col_t1, col_t2 = st.columns([1.6, 1.0])

    with col_t1:
        st.markdown("""<div class="card-box-header">📋 Benchmark Execution Summary</div>""", unsafe_allow_html=True)
        if df is not None and not df.empty:
            summary_df = df.copy()
            time_col = 'median_time' if 'median_time' in summary_df.columns else 'execution_time'
            summary_df = summary_df[['N', 'threads', 'schedule', 'chunk', time_col, 'speedup', 'efficiency']]
            summary_df.columns = ['Workload (N)', 'Threads', 'Schedule', 'Chunk', 'Time (s)', 'Speedup', 'Efficiency (%)']
            summary_df = summary_df.sort_values(by='Speedup', ascending=False).reset_index(drop=True)
            
            st.dataframe(
                summary_df.head(8).style.highlight_max(axis=0, subset=['Speedup'], color='rgba(16, 185, 129, 0.25)'),
                use_container_width=True,
                height=280
            )
        else:
            st.warning("No measured benchmark data present.")

    with col_t2:
        st.markdown("""<div class="card-box-header">🤖 AI Optimal Recommendation</div>""", unsafe_allow_html=True)
        if model is not None:
            rec_df = get_ai_recommendation(model, 80000000)
            best_rec = rec_df.iloc[0]

            st.markdown(f"""
            <div class="rec-box">
                <div class="rec-header">OPTIMAL CANDIDATE (N = 80M)</div>
                <div style="font-size: 20px; font-weight: 800; color: #F8FAFC;">
                    {int(best_rec['threads'])} Threads · {str(best_rec['schedule']).upper()}
                </div>
                <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">
                    Chunk Size: {int(best_rec['chunk'])}
                </div>
                <div style="font-size: 15px; color: #10B981; margin-top: 10px; font-weight: 700;">
                    Predicted Speedup: {best_rec['predicted_speedup']:.2f}×
                </div>
                <div style="font-size: 11px; color: #64748B; margin-top: 8px;">
                    Go to AI Optimizer to predict & validate with your own inputs.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model unavailable. Run 'make train' to train model.")

    st.markdown("""
    <div class="hw-note">
        ⚠️ Hardware Notice: Benchmark measurements reflect current host CPU multi-core capabilities and OS thread scheduler behavior.
    </div>
    """, unsafe_allow_html=True)
