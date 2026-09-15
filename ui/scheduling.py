# ui/scheduling.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ui.utils import (
    load_dataset,
    measure_median_time,
    save_experiment_record,
    ensure_binaries,
    SEQ_BIN,
    PAR_BIN
)

def render_scheduling_analysis():
    st.markdown("""
    <div class="page-header">
        <div class="title">⚡ OpenMP Scheduling Analysis</div>
        <div class="subtitle">Interactively configure and benchmark loop scheduling algorithms (Static, Dynamic, Guided) with live dynamic graph updates.</div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Schedule Policy Cards
    c_s1, c_s2, c_s3 = st.columns(3)

    c_s1.markdown("""
    <div class="card-box" style="border-top: 3px solid #38BDF8;">
        <div style="font-size: 15px; font-weight: 700; color: #38BDF8;">STATIC SCHEDULING</div>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
            Divides loop iterations into fixed chunks assigned round-robin at startup with zero lock overhead.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_s2.markdown("""
    <div class="card-box" style="border-top: 3px solid #818CF8;">
        <div style="font-size: 15px; font-weight: 700; color: #818CF8;">DYNAMIC SCHEDULING</div>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
            Chunks placed in shared queue; idle threads fetch work dynamically (susceptible to lock contention).
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_s3.markdown("""
    <div class="card-box" style="border-top: 3px solid #10B981;">
        <div style="font-size: 15px; font-weight: 700; color: #10B981;">GUIDED SCHEDULING</div>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
            Starts with large chunks shrinking exponentially down to chunk size, balancing dynamic load.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Interactive Control Panel for Live Demonstration
    st.markdown("""<div class="card-box-header">⚙️ Interactive Demonstration Controls</div>""", unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

    with col_ctrl1:
        selected_N = st.selectbox(
            "Workload Size (N)",
            options=[10000000, 20000000, 50000000, 80000000, 100000000],
            format_func=lambda x: f"{x//1000000}M Iterations ({x:,})",
            index=2
        )

    with col_ctrl2:
        selected_threads = st.selectbox("OpenMP Threads", options=[1, 2, 4, 8], index=3)

    with col_ctrl3:
        selected_schedules = st.multiselect(
            "Scheduling Policies",
            options=["static", "dynamic", "guided"],
            default=["static", "dynamic", "guided"]
        )

    with col_ctrl4:
        selected_chunks = st.multiselect(
            "Chunk Sizes",
            options=[100, 1000, 10000],
            default=[100, 1000, 10000]
        )

    run_demo = st.button("⚡ Run Live Demonstration & Generate Graphs", use_container_width=True, type="primary")

    if run_demo:
        if not selected_schedules or not selected_chunks:
            st.warning("Please select at least one scheduling policy and one chunk size.")
            return

        ensure_ok, msg = ensure_binaries()
        if not ensure_ok:
            st.error(msg)
            return

        total_runs = len(selected_schedules) * len(selected_chunks)
        progress_text = f"Executing C binaries live (Sequential baseline + {total_runs} parallel configurations)..."
        progress_bar = st.progress(0, text=progress_text)

        t_seq, _ = measure_median_time([SEQ_BIN, str(selected_N)])
        
        live_records = []
        completed = 0

        for sched in selected_schedules:
            for chunk in selected_chunks:
                completed += 1
                progress_bar.progress(completed / total_runs, text=f"Running {sched.upper()} (Chunk: {chunk})...")
                
                t_par, pi_val = measure_median_time([PAR_BIN, str(selected_N), str(selected_threads), sched, str(chunk)])
                if t_seq and t_par:
                    speedup = t_seq / t_par
                    efficiency = (speedup / selected_threads) * 100.0
                    
                    record = {
                        'N': selected_N,
                        'threads': selected_threads,
                        'schedule': sched,
                        'chunk': chunk,
                        'execution_time': round(t_par, 6),
                        'speedup': round(speedup, 2),
                        'efficiency': round(efficiency, 1)
                    }
                    live_records.append(record)
                    
                    save_experiment_record({
                        'N': selected_N,
                        'threads': selected_threads,
                        'schedule': sched,
                        'chunk': chunk,
                        'seq_time': round(t_seq, 6),
                        'par_time': round(t_par, 6),
                        'speedup': round(speedup, 2),
                        'efficiency': round(efficiency, 1)
                    })

        progress_bar.empty()
        
        if live_records:
            st.session_state['live_sched_df'] = pd.DataFrame(live_records)
            st.session_state['live_N'] = selected_N
            st.session_state['live_threads'] = selected_threads
            st.session_state['live_t_seq'] = t_seq

    # Determine whether to use Live Run results or Empirical Dataset fallback
    active_df = None
    is_live = False
    current_N = selected_N
    current_threads = selected_threads
    t_seq_baseline = None

    if 'live_sched_df' in st.session_state and not st.session_state['live_sched_df'].empty:
        active_df = st.session_state['live_sched_df'].copy()
        is_live = True
        current_N = st.session_state.get('live_N', selected_N)
        current_threads = st.session_state.get('live_threads', selected_threads)
        t_seq_baseline = st.session_state.get('live_t_seq', None)
    else:
        emp_df = load_dataset()
        if emp_df is not None and not emp_df.empty:
            sub = emp_df[(emp_df['N'] == selected_N) & (emp_df['threads'] == selected_threads)].copy()
            if not sub.empty:
                time_col = 'median_time' if 'median_time' in sub.columns else 'execution_time'
                sub['execution_time'] = sub[time_col]
                active_df = sub

    # Render Matrix & Dynamic Graphs
    if active_df is not None and not active_df.empty:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        mode_label = "⚡ LIVE MEASURED EXECUTION RESULTS" if is_live else "📊 EMPIRICAL DATASET RESULTS"
        st.markdown(f"""<div class="card-box-header">{mode_label} (N = {current_N:,}, {current_threads} Threads)</div>""", unsafe_allow_html=True)

        sorted_df = active_df.sort_values(by='speedup', ascending=False).reset_index(drop=True)
        sorted_df['Rank'] = range(1, len(sorted_df) + 1)
        
        display_df = sorted_df[['Rank', 'schedule', 'threads', 'chunk', 'execution_time', 'speedup', 'efficiency']].copy()
        display_df.columns = ['Rank', 'Schedule Policy', 'Threads', 'Chunk Size', 'Execution Time (s)', 'Speedup', 'Efficiency (%)']

        st.dataframe(
            display_df.style.highlight_max(axis=0, subset=['Speedup'], color='rgba(16, 185, 129, 0.25)'),
            use_container_width=True,
            height=240
        )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # 2 Dynamic Graphs
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("""<div class="card-box-header">1. Dynamic Measured Speedup by Chunk Size</div>""", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            fig.patch.set_facecolor('#131B2E')
            ax.set_facecolor('#090D16')

            pivot_df = sorted_df.pivot(index='chunk', columns='schedule', values='speedup')
            avail_cols = [c for c in ['static', 'dynamic', 'guided'] if c in pivot_df.columns]
            
            pivot_df[avail_cols].plot(kind='bar', ax=ax, color=['#38BDF8', '#818CF8', '#10B981'], width=0.6)
            ax.set_title(f"Speedup across Configurations (Higher is Better)", color='#F8FAFC', fontsize=10, pad=8)
            ax.set_xlabel("Chunk Size", color='#94A3B8', fontsize=8)
            ax.set_ylabel("Speedup (S = T1 / Tp)", color='#94A3B8', fontsize=8)
            ax.tick_params(colors='#94A3B8', labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.15, color='#94A3B8')
            ax.legend(facecolor='#131B2E', edgecolor='#1E293B', labelcolor='#F8FAFC', fontsize=8)
            plt.xticks(rotation=0)
            for spine in ax.spines.values(): spine.set_color('#1E293B')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_g2:
            st.markdown("""<div class="card-box-header">2. Dynamic Execution Time (Lower is Better)</div>""", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            fig.patch.set_facecolor('#131B2E')
            ax.set_facecolor('#090D16')

            time_pivot = sorted_df.pivot(index='chunk', columns='schedule', values='execution_time')
            avail_cols = [c for c in ['static', 'dynamic', 'guided'] if c in time_pivot.columns]
            
            time_pivot[avail_cols].plot(kind='bar', ax=ax, color=['#38BDF8', '#818CF8', '#10B981'], width=0.6)
            ax.set_title(f"Execution Time in Seconds (Lower is Better)", color='#F8FAFC', fontsize=10, pad=8)
            ax.set_xlabel("Chunk Size", color='#94A3B8', fontsize=8)
            ax.set_ylabel("Execution Time (s)", color='#94A3B8', fontsize=8)
            ax.tick_params(colors='#94A3B8', labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.15, color='#94A3B8')
            ax.legend(facecolor='#131B2E', edgecolor='#1E293B', labelcolor='#F8FAFC', fontsize=8)
            plt.xticks(rotation=0)
            for spine in ax.spines.values(): spine.set_color('#1E293B')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Dynamic Performance Insight Panel
        best_row = sorted_df.iloc[0]
        worst_row = sorted_df.iloc[-1]
        
        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid #10B981; margin-top: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #10B981; text-transform: uppercase;">DYNAMIC DEMONSTRATION INSIGHT</div>
            <div style="font-size: 13px; color: #F8FAFC; margin-top: 6px; line-height: 1.5;">
                🏆 <b>Best Configuration:</b> <b>{str(best_row['schedule']).upper()}</b> scheduling with <b>chunk size {int(best_row['chunk'])}</b> achieved peak speedup of <b>{best_row['speedup']:.2f}x</b> ({best_row['execution_time']:.6f}s).<br>
                ⚠️ <b>Lowest Performance:</b> <b>{str(worst_row['schedule']).upper()}</b> scheduling with <b>chunk size {int(worst_row['chunk'])}</b> ran in {worst_row['execution_time']:.6f}s ({worst_row['speedup']:.2f}x speedup).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No scheduling data available. Configure parameters above and click 'Run Live Demonstration'.")
