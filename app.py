# app.py
# ------------------------------------------------------------------
# Project: AI-Assisted Parallel Numerical Computation & Scheduling Using C/OpenMP
# Description: Academic Streamlit UI for OpenMP Pi computations,
#              parallel metrics evaluation, live schedule comparisons, and AI predictions.
# ------------------------------------------------------------------

import os
import sys
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import joblib

os.environ['MPLCONFIGDIR'] = '/tmp'

st.set_page_config(
    page_title="AI-Assisted OpenMP Parallel Computing",
    page_icon="⚡",
    layout="wide"
)

# Clean, high-contrast academic styling (No animations)
st.markdown("""
<style>
    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #38BDF8;
        margin-bottom: 12px;
    }
    .academic-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

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
    best = df_cand.sort_values(by='predicted_speedup', ascending=False).iloc[0]
    return best

# Main Layout Header
st.markdown('<div class="main-title">AI-Assisted Parallel Numerical Computation & Scheduling</div>', unsafe_allow_html=True)

seq_bin = os.path.join("bin", "sequential_pi")
par_bin = os.path.join("bin", "parallel_pi")
model_path = os.path.join("model", "speedup_model.pkl")

# Verify prerequisites
if not os.path.exists(seq_bin) or not os.path.exists(par_bin):
    st.warning("Compiled C binaries missing. Building executables via 'make all'...")
    subprocess.run(["make", "all"], check=True)

model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("AI model 'model/speedup_model.pkl' not found. Run 'make train' to train model.")

# Sidebar Controls
st.sidebar.header("Experimental Parameters")
input_N = st.sidebar.number_input("Workload Size (N steps)", min_value=100000, max_value=500000000, value=100000000, step=10000000)
threads = st.sidebar.selectbox("Number of OpenMP Threads", options=[1, 2, 4, 8, 16], index=3)
schedule = st.sidebar.selectbox("OpenMP Scheduling Strategy", options=["static", "dynamic", "guided"], index=0)
chunk_size = st.sidebar.selectbox("Chunk Size", options=[100, 1000, 10000], index=1)

run_button = st.sidebar.button("Run Parallel Computation & AI Prediction", type="primary")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Run & AI Prediction",
    "⚡ Compare All Schedules",
    "Empirical Dataset",
    "Performance Graphs"
])

with tab1:
    st.subheader("Live Execution & Performance Metrics")
    
    if run_button or 'last_run' not in st.session_state:
        with st.spinner("Executing compiled C OpenMP binaries..."):
            t_seq, pi_seq = run_c_binary([seq_bin, str(input_N)])
            t_par, pi_par = run_c_binary([par_bin, str(input_N), str(threads), schedule, str(chunk_size)])
            
            if t_seq and t_par:
                speedup = t_seq / t_par
                efficiency = (speedup / threads) * 100.0
                
                pred_speedup = None
                best_config = None
                if model:
                    user_df = pd.DataFrame([{'N': input_N, 'threads': threads, 'schedule': schedule, 'chunk': chunk_size}])
                    pred_speedup = float(model.predict(user_df)[0])
                    best_config = get_ai_recommendation(model, input_N)
                
                st.session_state['last_run'] = {
                    'N': input_N, 'threads': threads, 'schedule': schedule, 'chunk': chunk_size,
                    't_seq': t_seq, 't_par': t_par, 'pi_seq': pi_seq, 'pi_par': pi_par,
                    'speedup': speedup, 'efficiency': efficiency,
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
            st.subheader("Numerical Calculation Verification")
            st.write(f"**Target Workload (N)**: `{run['N']:,}` steps")
            st.write(f"**Sequential Calculated $\\pi$**: `{run['pi_seq']:.15f}`")
            st.write(f"**Parallel Calculated $\\pi$**: `{run['pi_par']:.15f}`")
            exact_pi = 3.141592653589793
            st.write(f"**Absolute Error vs Exact $\\pi$**: `{abs(run['pi_par'] - exact_pi):.15e}`")

        with col_right:
            st.subheader("AI Performance Model Prediction")
            if run['pred_speedup'] is not None:
                st.write(f"**AI Predicted Speedup**: `{run['pred_speedup']:.2f}x`")
                st.write(f"**Measured Actual Speedup**: `{run['speedup']:.2f}x`")
                err = abs(run['pred_speedup'] - run['speedup'])
                pct_err = (err / run['speedup']) * 100.0
                st.write(f"**Prediction Error**: `{err:.4f}` ({pct_err:.2f}%)")
            
            if run['best_config'] is not None:
                b = run['best_config']
                st.success(f"**AI Recommended Optimal Setting for N = {run['N']:,}**:\n"
                           f"- Threads: **{int(b['threads'])}**\n"
                           f"- Schedule: **{b['schedule'].upper()}**\n"
                           f"- Chunk Size: **{int(b['chunk'])}**\n"
                           f"- Max Predicted Speedup: **{b['predicted_speedup']:.2f}x**")

with tab2:
    st.subheader(f"⚡ Live Comparison of All OpenMP Scheduling Algorithms (N = {input_N:,}, {threads} Threads)")
    st.write("Executes **Static**, **Dynamic**, and **Guided** OpenMP schedules across chunk sizes (100, 1000, 10000) live and ranks them by execution time.")
    
    if st.button("Run All Scheduling Algorithms Now", type="primary"):
        with st.spinner("Executing sequential baseline and 9 parallel schedule/chunk combinations..."):
            t_seq, _ = run_c_binary([seq_bin, str(input_N)])
            
            sched_list = ["static", "dynamic", "guided"]
            chunk_list = [100, 1000, 10000]
            
            records = []
            for s in sched_list:
                for c in chunk_list:
                    t_min = 1e9
                    for _ in range(2):
                        t_p, _ = run_c_binary([par_bin, str(input_N), str(threads), s, str(c)])
                        if t_p and t_p < t_min:
                            t_min = t_p
                    
                    sp = t_seq / t_min
                    eff = (sp / threads) * 100.0
                    records.append({
                        'Schedule': s.upper(),
                        'Chunk Size': c,
                        'Execution Time (s)': round(t_min, 6),
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
            ### 🏆 WINNER: **{best['Schedule']}** (Chunk: {best['Chunk Size']})
            - **Execution Time**: `{best['Execution Time (s)']} seconds`
            - **Speedup**: **`{best['Speedup']}x` faster** than sequential (`{t_seq:.6f} s`)
            - **Parallel Efficiency**: **`{best['Efficiency (%)']}%`**
            """)
        with col_w2:
            st.error(f"""
            ### ⚠️ SLOWEST: **{worst['Schedule']}** (Chunk: {worst['Chunk Size']})
            - **Execution Time**: `{worst['Execution Time (s)']} seconds` (**`{worst['Execution Time (s)'] / best['Execution Time (s)']:.2f}x` slower**)
            - **Speedup**: `{worst['Speedup']}x`
            - **Parallel Efficiency**: `{worst['Efficiency (%)']}%`
            """)

        st.subheader("Ranked Performance Summary Table")
        st.dataframe(df_comp, use_container_width=True)

        st.subheader("Execution Time Comparison Chart (Lower is Better)")
        st.bar_chart(data=df_comp, x="Schedule", y="Execution Time (s)", color="Chunk Size", use_container_width=True)

        with st.expander("📖 Why is one schedule better than the others?"):
            st.markdown("""
            ### 1. GUIDED Schedule (Best for dynamic/large workloads)
            - **Mechanism**: Starts with large chunk sizes and exponentially shrinks them down to `chunk_size` near loop completion.
            - **Why it wins**: Minimizes task-queue locking overhead at the start while providing dynamic load balancing near the end.

            ### 2. STATIC Schedule (Best for uniform workloads)
            - **Mechanism**: Divides iterations into equal blocks at start time with zero runtime queue overhead.
            - **Why it performs well**: Since each iteration of midpoint integration takes identical CPU cycles, static scheduling has zero lock contention.

            ### 3. DYNAMIC Schedule (Worst when chunk size is small)
            - **Mechanism**: Chunks are placed in a shared queue. Threads grab chunks dynamically as they finish.
            - **Why small chunks fail**: With `chunk=100` and $N=100\text{M}$, there are $1,000,000$ queue lock requests. Mutex synchronization overhead completely destroys parallel speedup!
            """)

with tab3:
    st.subheader("Empirical Benchmark Dataset (data/performance.csv)")
    csv_path = os.path.join("data", "performance.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.write(f"Total benchmark records: `{len(df)}`")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No benchmark dataset found. Run `make run-benchmark` to generate empirical data.")

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
