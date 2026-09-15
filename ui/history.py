# ui/history.py
import streamlit as st
import pandas as pd

from ui.utils import load_experiment_history, load_dataset

def render_experiment_history():
    st.markdown("""
    <div class="page-header">
        <div class="title">📜 Experiment History</div>
        <div class="subtitle">Complete audit trail and history of executed OpenMP parallel experiments.</div>
    </div>
    """, unsafe_allow_html=True)

    history = load_experiment_history()

    if not history:
        # Fallback to dataset if history JSON is empty
        df = load_dataset()
        if df is not None and not df.empty:
            time_col = 'median_time' if 'median_time' in df.columns else 'execution_time'
            history = []
            for _, row in df.iterrows():
                history.append({
                    'timestamp': 'Empirical Dataset',
                    'N': int(row['N']),
                    'threads': int(row['threads']),
                    'schedule': str(row['schedule']),
                    'chunk': int(row['chunk']),
                    'seq_time': None,
                    'par_time': round(row[time_col], 6),
                    'speedup': round(row['speedup'], 2),
                    'efficiency': round(row['efficiency'], 1)
                })

    if not history:
        st.info("No experiment history recorded yet. Run experiments in the Benchmark Lab or AI Optimizer.")
        return

    df_hist = pd.DataFrame(history)

    # Summary Banner
    best_sp = df_hist['speedup'].max()
    worst_sp = df_hist['speedup'].min()
    avg_sp = df_hist['speedup'].mean()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Recorded Experiments", len(df_hist))
    col_m2.metric("Best Measured Speedup", f"{best_sp:.2f}x")
    col_m3.metric("Worst Measured Speedup", f"{worst_sp:.2f}x")
    col_m4.metric("Average Speedup", f"{avg_sp:.2f}x")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter Controls
    st.markdown("""<div class="card-box-header">🔍 Filter Experiment Audit Trail</div>""", unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        workload_filter = st.multiselect("Workload (N)", options=sorted(df_hist['N'].unique()), default=[])
    with col_f2:
        threads_filter = st.multiselect("Threads", options=sorted(df_hist['threads'].unique()), default=[])
    with col_f3:
        sched_filter = st.multiselect("Schedule Policy", options=sorted(df_hist['schedule'].unique()), default=[])

    filtered_df = df_hist.copy()
    if workload_filter:
        filtered_df = filtered_df[filtered_df['N'].isin(workload_filter)]
    if threads_filter:
        filtered_df = filtered_df[filtered_df['threads'].isin(threads_filter)]
    if sched_filter:
        filtered_df = filtered_df[filtered_df['schedule'].isin(sched_filter)]

    # Format Table
    display_df = filtered_df[['timestamp', 'N', 'threads', 'schedule', 'chunk', 'par_time', 'speedup', 'efficiency']].copy()
    display_df.columns = ['Timestamp', 'Workload (N)', 'Threads', 'Schedule', 'Chunk', 'Parallel Time (s)', 'Speedup', 'Efficiency (%)']

    st.dataframe(
        display_df.style.highlight_max(axis=0, subset=['Speedup'], color='rgba(52, 211, 153, 0.3)'),
        use_container_width=True,
        height=400
    )
