# app.py
# ------------------------------------------------------------------
# Project: AI-Assisted OpenMP Scheduling for Parallel Numerical Computation
# Description: Clean, concise academic Streamlit UI for OpenMP Pi computations,
#              metrics evaluation, live schedule comparisons, and AI predictions.
# ------------------------------------------------------------------

import os
import sys
import platform
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import joblib

os.environ['MPLCONFIGDIR'] = '/tmp'

st.set_page_config(
    page_title="AI-Assisted OpenMP Computing",
    page_icon="⚡",
    layout="wide"
)

# Clean academic CSS
st.markdown("""
<style>
    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 13px;
        color: #94A3B8;
        margin-bottom: 16px;
    }
    .hw-note {
        font-size: 11px;
        color: #F59E0B;
        background: rgba(245, 158, 11, 0.1);
        border-left: 3px solid #F59E0B;
        padding: 6px 10px;
        border-radius: 4px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

WARMUP_RUNS = 1
MEASURED_RUNS = 5

def run_c_binary(cmd_list):
    try:
        res = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        time_val, pi_val = None, None
        for line in res.stdout.splitlines():
            if "Execution Time (s)" in line:
                time_val = float(line.split(":")[-1].strip())
            elif "Calculated Pi" in line:
                pi_val = float(line.split(":")[-1].strip())
        return time_val, pi_val
    except Exception as e:
        st.error(f"Error running binary {cmd_list}: {e}")
        return None, None

def measure_median_time(cmd_list):
    for _ in range(WARMUP_RUNS):
        run_c_binary(cmd_list)
    
    times = []
    pi_val = None
    for _ in range(MEASURED_RUNS):
        t, p = run_c_binary(cmd_list)
        if t is not None:
            times.append(t)
            pi_val = p
            
    if not times:
        return None, None
    return float(np.median(times)), pi_val

def get_ai_recommendation(model, N):
    schedules = ['static', 'dynamic', 'guided']
    threads_list = [2, 4, 8]
    chunks = [100, 1000, 10000]

    candidates = []
    for sched in schedules:
        for t in threads_list:
            for c in chunks:
                candidates.append({'N': N, 'threads': t, 'schedule': sched, 'chunk': c})

    df_cand = pd.DataFrame(candidates)
    df_cand['predicted_speedup'] = model.predict(df_cand)
    sorted_df = df_cand.sort_values(by='predicted_speedup', ascending=False).reset_index(drop=True)
    return sorted_df

# Header
st.markdown('<div class="main-title">AI-Assisted OpenMP Parallel Computation & Scheduling</div>', unsafe_allow_html=True)

seq_bin = os.path.join("bin", "sequential_pi")
par_bin = os.path.join("bin", "parallel_pi")
model_path = os.path.join("model", "speedup_model.pkl")

# Verify binaries
if not os.path.exists(seq_bin) or not os.path.exists(par_bin):
    st.warning("Building C binaries via 'make all'...")
    subprocess.run(["make", "all"], check=True)

model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model 'model/speedup_model.pkl' missing. Run 'make train'.")

# Sidebar Controls
st.sidebar.header("Experimental Parameters")
input_N = st.sidebar.number_input("Workload Size (N steps)", min_value=100000, max_value=500000000, value=50000000, step=10000000)

threads = st.sidebar.selectbox("Number of OpenMP Threads", options=[1, 2, 4, 8], index=3)
schedule = st.sidebar.selectbox("OpenMP Scheduling Strategy", options=["static", "dynamic", "guided"], index=0)
chunk_size = st.sidebar.selectbox("Chunk Size", options=[100, 1000, 10000], index=1)

run_button = st.sidebar.button("Run Experiment & AI Prediction", type="primary")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Run & AI Prediction",
    "⚡ Compare All Schedules",
    "Empirical Dataset",
    "Performance Graphs"
])

# ------------------------------------------------------------------
# TAB 1: SINGLE RUN & AI PREDICTION
# ------------------------------------------------------------------
with tab1:
    st.subheader("Live Execution & Performance Metrics")
    
    if run_button or 'last_run' not in st.session_state:
        with st.spinner("Executing compiled C binaries (1 warmup + 5 measured runs)..."):
            t_seq, pi_seq = measure_median_time([seq_bin, str(input_N)])
            t_par, pi_par = measure_median_time([par_bin, str(input_N), str(threads), schedule, str(chunk_size)])
            
            if t_seq and t_par:
                speedup = t_seq / t_par
                efficiency = (speedup / threads) * 100.0
                exact_pi = 3.14159265358979323846
                pi_error = abs(pi_par - exact_pi)
                
                pred_speedup = None
                best_config = None
                if model:
                    user_df = pd.DataFrame([{'N': input_N, 'threads': threads, 'schedule': schedule, 'chunk': chunk_size}])
                    pred_speedup = float(model.predict(user_df)[0])
                    best_config = get_ai_recommendation(model, input_N).iloc[0]
                
                st.session_state['last_run'] = {
                    'N': input_N, 'threads': threads, 'schedule': schedule, 'chunk': chunk_size,
                    't_seq': t_seq, 't_par': t_par, 'pi_seq': pi_seq, 'pi_par': pi_par,
                    'pi_error': pi_error, 'speedup': speedup, 'efficiency': efficiency,
                    'pred_speedup': pred_speedup, 'best_config': best_config
                }

    if 'last_run' in st.session_state:
        run = st.session_state['last_run']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sequential Time", f"{run['t_seq']:.6f} s")
        c2.metric("Parallel Time", f"{run['t_par']:.6f} s")
        c3.metric("Measured Speedup", f"{run['speedup']:.2f}x")
        c4.metric("Parallel Efficiency", f"{run['efficiency']:.1f}%")

        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Numerical Verification")
            st.write(f"**Workload (N)**: `{run['N']:,}` steps")
            st.write(f"**Sequential Calculated $\\pi$**: `{run['pi_seq']:.15f}`")
            st.write(f"**Parallel Calculated $\\pi$**: `{run['pi_par']:.15f}`")
            st.write(f"**Floating-Point Error**: `{run['pi_error']:.15e}`")

        with col_right:
            st.subheader("AI Performance Prediction")
            if run['pred_speedup'] is not None:
                st.write(f"**AI Predicted Speedup**: `{run['pred_speedup']:.2f}x`")
                st.write(f"**Measured Actual Speedup**: `{run['speedup']:.2f}x`")
                err = abs(run['pred_speedup'] - run['speedup'])
                pct_err = (err / run['speedup']) * 100.0
                st.write(f"**Prediction Error**: `{err:.4f}` ({pct_err:.2f}%)")
            
            if run['best_config'] is not None:
                b = run['best_config']
                st.success(f"**Recommended Configuration for N = {run['N']:,}**:\n"
                           f"- Threads: **{int(b['threads'])}**\n"
                           f"- Schedule: **{b['schedule'].upper()}**\n"
                           f"- Chunk Size: **{int(b['chunk'])}**\n"
                           f"- Highest Predicted Speedup: **{b['predicted_speedup']:.2f}x**")

    # System Info
    st.markdown("---")
    col_hw1, col_hw2, col_hw3, col_hw4 = st.columns(4)
    col_hw1.caption(f"**CPU Cores**: {os.cpu_count()} Logical")
    col_hw2.caption(f"**Supported Threads**: 1, 2, 4, 8")
    col_hw3.caption(f"**Python**: {sys.version.split()[0]}")
    col_hw4.caption(f"**OS**: {platform.system()} ({platform.machine()})")
    st.markdown('<div class="hw-note">⚠️ Note: Performance measurements are hardware-dependent for the current host machine.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# TAB 2: SCHEDULE COMPARISON
# ------------------------------------------------------------------
with tab2:
    st.subheader(f"⚡ Live Comparison of All OpenMP Scheduling Strategies (N = {input_N:,}, {threads} Threads)")
    
    if st.button("Run All Scheduling Algorithms Now", type="primary"):
        with st.spinner("Executing sequential baseline and 9 parallel configurations..."):
            t_seq, _ = measure_median_time([seq_bin, str(input_N)])
            
            sched_list = ["static", "dynamic", "guided"]
            chunk_list = [100, 1000, 10000]
            
            records = []
            for s in sched_list:
                for c in chunk_list:
                    t_med, _ = measure_median_time([par_bin, str(input_N), str(threads), s, str(c)])
                    if t_med:
                        sp = t_seq / t_med
                        eff = (sp / threads) * 100.0
                        records.append({
                            'Schedule': s.upper(),
                            'Chunk Size': c,
                            'Execution Time (s)': round(t_med, 6),
                            'Speedup': round(sp, 2),
                            'Efficiency (%)': round(eff, 1)
                        })
            
            res_df = pd.DataFrame(records).sort_values(by='Execution Time (s)').reset_index(drop=True)
            st.session_state['sched_comp'] = {
                't_seq': t_seq,
                'df': res_df,
                'N': input_N,
                'threads': threads
            }

    if 'sched_comp' in st.session_state:
        sc = st.session_state['sched_comp']
        df_comp = sc['df']
        t_seq = sc['t_seq']
        
        best = df_comp.iloc[0]
        worst = df_comp.iloc[-1]
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.success(f"""
            ### 🏆 BEST SCHEDULE: **{best['Schedule']}** (Chunk: {best['Chunk Size']})
            - **Time**: `{best['Execution Time (s)']} s`
            - **Speedup**: **`{best['Speedup']}x`** vs sequential baseline
            - **Efficiency**: **`{best['Efficiency (%)']}%`**
            """)
        with col_w2:
            st.error(f"""
            ### ⚠️ SLOWEST SCHEDULE: **{worst['Schedule']}** (Chunk: {worst['Chunk Size']})
            - **Time**: `{worst['Execution Time (s)']} s` (**`{worst['Execution Time (s)'] / best['Execution Time (s)']:.2f}x` slower**)
            - **Speedup**: `{worst['Speedup']}x`
            - **Efficiency**: `{worst['Efficiency (%)']}%`
            """)

        st.subheader("Ranked Performance Summary Table")
        st.dataframe(df_comp, use_container_width=True)

        st.subheader("Execution Time Comparison Chart (Lower is Better)")
        st.bar_chart(data=df_comp, x="Schedule", y="Execution Time (s)", color="Chunk Size", use_container_width=True)

# ------------------------------------------------------------------
# TAB 3: EMPIRICAL DATASET
# ------------------------------------------------------------------
with tab3:
    st.subheader("Empirical Benchmark Dataset (data/performance.csv)")
    csv_path = os.path.join("data", "performance.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.write(f"Total benchmark records: `{len(df)}`")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No benchmark dataset found. Run `make run-benchmark` to generate empirical data.")

# ------------------------------------------------------------------
# TAB 4: PERFORMANCE GRAPHS
# ------------------------------------------------------------------
with tab4:
    st.subheader("Performance Analysis Plots")
    res_dir = "results"
    plots = [
        ("Execution Time vs Threads", "execution_time_vs_threads.png"),
        ("Speedup vs Threads", "speedup_vs_threads.png"),
        ("Efficiency vs Threads", "efficiency_vs_threads.png"),
        ("OpenMP Schedule Policy Comparison", "schedule_comparison.png"),
        ("Workload Scaling", "workload_scaling.png"),
        ("AI Predicted vs Actual Speedup", "predicted_vs_actual.png")
    ]
    
    p1, p2 = st.columns(2)
    for idx, (title, img_name) in enumerate(plots):
        img_path = os.path.join(res_dir, img_name)
        target_col = p1 if idx % 2 == 0 else p2
        with target_col:
            st.write(f"**{title}**")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info(f"Plot {img_name} not generated. Run `make visualize`.")
