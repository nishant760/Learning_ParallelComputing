# ui/benchmark.py
import streamlit as st
import pandas as pd
import time

from ui.utils import (
    ensure_binaries,
    measure_median_time,
    save_experiment_record,
    load_model,
    SEQ_BIN,
    PAR_BIN
)

def render_benchmark_lab():
    st.markdown("""
    <div class="page-header">
        <div class="title">🔬 Benchmark Laboratory</div>
        <div class="subtitle">Run controlled sequential and OpenMP parallel numerical experiments.</div>
    </div>
    """, unsafe_allow_html=True)

    ensure_ok, msg = ensure_binaries()
    if not ensure_ok:
        st.error(msg)
        return

    # Control Panel Layout
    st.markdown("""<div class="card-box-header">⚙️ Experimental Parameters & Configuration</div>""", unsafe_allow_html=True)

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    with col_c1:
        workload_choice = st.selectbox(
            "Workload Size (N)",
            options=["10M (10,000,000)", "20M (20,000,000)", "50M (50,000,000)", "80M (80,000,000)", "100M (100,000,000)", "Custom"],
            index=2
        )
        if workload_choice == "Custom":
            N = st.number_input("Custom N Steps", min_value=100000, max_value=500000000, value=50000000, step=10000000)
        else:
            n_map = {
                "10M (10,000,000)": 10000000,
                "20M (20,000,000)": 20000000,
                "50M (50,000,000)": 50000000,
                "80M (80,000,000)": 80000000,
                "100M (100,000,000)": 100000000
            }
            N = n_map[workload_choice]

    with col_c2:
        threads = st.selectbox("OpenMP Threads", options=[1, 2, 4, 8], index=3)

    with col_c3:
        schedule = st.selectbox("Scheduling Strategy", options=["static", "dynamic", "guided"], index=0)

    with col_c4:
        chunk_size = st.selectbox("Chunk Size", options=[100, 1000, 10000], index=1)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_bench = st.button("🚀 Run Experiment", use_container_width=True, type="primary")
    with col_btn2:
        if st.button("🔄 Reset Configuration", use_container_width=False):
            st.session_state.pop('bench_run', None)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if run_bench:
        status_box = st.empty()
        
        status_box.markdown("""
        <div class="card-box">
            <div style="font-size: 14px; font-weight: 700; color: #38BDF8; margin-bottom: 10px;">BENCHMARK RUNNING</div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="color: #34D399;">● OpenMP Compilation — Complete</div>
                <div style="color: #F59E0B;">● Sequential Execution — Running (1 Warm-up + 5 Measured Runs)...</div>
                <div style="color: #64748B;">○ Parallel Execution — Waiting</div>
                <div style="color: #64748B;">○ Performance Analysis — Waiting</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        t_seq, pi_seq = measure_median_time([SEQ_BIN, str(N)])

        status_box.markdown(f"""
        <div class="card-box">
            <div style="font-size: 14px; font-weight: 700; color: #38BDF8; margin-bottom: 10px;">BENCHMARK RUNNING</div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="color: #34D399;">● OpenMP Compilation — Complete</div>
                <div style="color: #34D399;">● Sequential Execution — Complete ({t_seq:.6f}s)</div>
                <div style="color: #F59E0B;">● Parallel Execution — Running (OpenMP {threads} Threads, {schedule.upper()} schedule)...</div>
                <div style="color: #64748B;">○ Performance Analysis — Waiting</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        t_par, pi_par = measure_median_time([PAR_BIN, str(N), str(threads), schedule, str(chunk_size)])

        status_box.markdown(f"""
        <div class="card-box">
            <div style="font-size: 14px; font-weight: 700; color: #38BDF8; margin-bottom: 10px;">BENCHMARK COMPLETE</div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="color: #34D399;">● OpenMP Compilation — Complete</div>
                <div style="color: #34D399;">● Sequential Execution — Complete ({t_seq:.6f}s)</div>
                <div style="color: #34D399;">● Parallel Execution — Complete ({t_par:.6f}s)</div>
                <div style="color: #34D399;">● Performance Analysis — Complete</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if t_seq and t_par:
            speedup = t_seq / t_par
            efficiency = (speedup / threads) * 100.0
            exact_pi = 3.14159265358979323846
            pi_error = abs(pi_par - exact_pi)

            st.session_state['bench_run'] = {
                'N': N, 'threads': threads, 'schedule': schedule, 'chunk': chunk_size,
                't_seq': t_seq, 't_par': t_par, 'pi_seq': pi_seq, 'pi_par': pi_par,
                'speedup': speedup, 'efficiency': efficiency, 'pi_error': pi_error
            }

            save_experiment_record({
                'N': N, 'threads': threads, 'schedule': schedule, 'chunk': chunk_size,
                'seq_time': round(t_seq, 6), 'par_time': round(t_par, 6),
                'speedup': round(speedup, 2), 'efficiency': round(efficiency, 1)
            })

    if 'bench_run' in st.session_state:
        run = st.session_state['bench_run']

        st.markdown("""<div class="card-box-header">📊 Measured Benchmark Results</div>""", unsafe_allow_html=True)
        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)

        rc1.metric("Sequential Time", f"{run['t_seq']:.6f} s")
        rc2.metric("Parallel Time", f"{run['t_par']:.6f} s")
        rc3.metric("Measured Speedup", f"{run['speedup']:.2f}x")
        rc4.metric("Parallel Efficiency", f"{run['efficiency']:.1f}%")
        rc5.metric("Calculated Pi", f"{run['pi_par']:.10f}")
        rc6.metric("Floating-Pt Error", f"{run['pi_error']:.2e}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-Side Visual Comparison
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown(f"""
            <div class="card-box" style="border-left: 4px solid #818CF8;">
                <div style="font-size: 13px; font-weight: 700; color: #818CF8; text-transform: uppercase;">Sequential Baseline (1 Core)</div>
                <div style="font-size: 32px; font-weight: 800; color: #F8FAFC; margin: 10px 0;">
                    {run['t_seq']:.6f} <span style="font-size: 16px; color: #94A3B8;">sec</span>
                </div>
                <div style="font-size: 12px; color: #94A3B8;">
                    Calculated π: <code>{run['pi_seq']:.15f}</code>
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">
                    Workload N: <code>{run['N']:,} steps</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_v2:
            st.markdown(f"""
            <div class="card-box" style="border-left: 4px solid #34D399;">
                <div style="font-size: 13px; font-weight: 700; color: #34D399; text-transform: uppercase;">OpenMP Parallel ({run['threads']} Threads)</div>
                <div style="font-size: 32px; font-weight: 800; color: #F8FAFC; margin: 10px 0;">
                    {run['t_par']:.6f} <span style="font-size: 16px; color: #94A3B8;">sec</span>
                    <span style="font-size: 18px; color: #34D399; font-weight: 700; margin-left: 12px;">({run['speedup']:.2f}x Faster)</span>
                </div>
                <div style="font-size: 12px; color: #94A3B8;">
                    Calculated π: <code>{run['pi_par']:.15f}</code>
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">
                    Schedule: <code>{run['schedule'].upper()} (Chunk: {run['chunk']})</code> | Efficiency: <code>{run['efficiency']:.1f}%</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
