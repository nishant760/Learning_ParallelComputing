# ui/benchmark.py
import streamlit as st
import pandas as pd

from ui.utils import (
    ensure_binaries,
    measure_median_time,
    save_experiment_record,
    SEQ_BIN,
    PAR_BIN
)

def render_benchmark_lab():
    st.markdown("""
    <div class="page-header">
        <div class="title">🔬 Benchmark Laboratory</div>
        <div class="subtitle">
            Run a targeted single-configuration experiment (Workload N, Threads, Schedule, Chunk Size)
            to measure exact timing, speedup, and numerical precision (Calculated π vs True π).
        </div>
    </div>
    """, unsafe_allow_html=True)

    ensure_ok, msg = ensure_binaries()
    if not ensure_ok:
        st.error(f"Build error: {msg}")
        return

    # Control Panel
    st.markdown("""<div class="card-box-header">⚙️ Target Configuration</div>""", unsafe_allow_html=True)

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    with col_c1:
        N = st.selectbox(
            "Workload Size (N)",
            options=[10_000_000, 20_000_000, 50_000_000, 80_000_000, 100_000_000],
            format_func=lambda x: f"{x // 1_000_000}M ({x:,})",
            index=2
        )

    with col_c2:
        threads = st.selectbox("OpenMP Threads", options=[1, 2, 4, 8], index=2)

    with col_c3:
        schedule = st.selectbox("Scheduling Strategy", options=["static", "dynamic", "guided"], index=0)

    with col_c4:
        chunk_size = st.selectbox("Chunk Size", options=[100, 1000, 10000], index=1)

    col_btn1, col_btn2 = st.columns([1.5, 4])
    with col_btn1:
        run_bench = st.button("🚀 Run Benchmark", use_container_width=True, type="primary")
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=False):
            st.session_state.pop('bench_run', None)
            st.rerun()

    if run_bench:
        with st.spinner("Measuring sequential baseline (1 warm-up + 5 measured runs)..."):
            t_seq, pi_seq = measure_median_time([SEQ_BIN, str(N)])

        if not t_seq:
            st.error("Failed to execute sequential C binary. Ensure binaries are compiled.")
            return

        with st.spinner(f"Measuring OpenMP parallel execution ({threads} threads, {schedule.upper()} schedule, chunk {chunk_size})..."):
            t_par, pi_par = measure_median_time([PAR_BIN, str(N), str(threads), schedule, str(chunk_size)])

        if not t_par:
            st.error("Failed to execute parallel C binary.")
            return

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

    # Display results
    if 'bench_run' in st.session_state:
        run = st.session_state['bench_run']

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-box-header">📊 Measured Results — N = {run['N']:,} · {run['threads']} Threads · {run['schedule'].upper()} (Chunk {run['chunk']})</div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sequential Time", f"{run['t_seq']:.4f}s")
        m2.metric("Parallel Time", f"{run['t_par']:.4f}s")
        m3.metric("Speedup", f"{run['speedup']:.2f}×")
        m4.metric("Parallel Efficiency", f"{run['efficiency']:.1f}%")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # Side-by-Side Comparison
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown(f"""
            <div class="card-box" style="border-left: 4px solid #818CF8; padding: 18px 20px;">
                <div style="font-size: 11px; font-weight: 700; color: #818CF8; text-transform: uppercase; letter-spacing: 1px;">
                    Sequential Baseline (1 Thread)
                </div>
                <div style="font-size: 26px; font-weight: 800; color: #F8FAFC; margin-top: 6px;">
                    {run['t_seq']:.4f} <span style="font-size: 14px; color: #94A3B8;">seconds</span>
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 10px;">
                    Calculated π: <code style="color:#F8FAFC;">{run['pi_seq']:.10f}</code>
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">
                    Workload: <code>{run['N']:,} steps</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_right:
            st.markdown(f"""
            <div class="card-box" style="border-left: 4px solid #10B981; padding: 18px 20px;">
                <div style="font-size: 11px; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 1px;">
                    OpenMP Parallel ({run['threads']} Threads)
                </div>
                <div style="font-size: 26px; font-weight: 800; color: #F8FAFC; margin-top: 6px;">
                    {run['t_par']:.4f} <span style="font-size: 14px; color: #94A3B8;">seconds</span>
                    <span style="font-size: 16px; color: #10B981; font-weight: 700; margin-left: 8px;">({run['speedup']:.2f}× Speedup)</span>
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 10px;">
                    Calculated π: <code style="color:#F8FAFC;">{run['pi_par']:.10f}</code> (Error: {run['pi_error']:.2e})
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">
                    Policy: <code>{run['schedule'].upper()}</code> | Chunk: <code>{run['chunk']}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card-box" style="text-align:center; padding: 36px 20px; color:#64748B;">
            <div style="font-size: 32px; margin-bottom: 6px;">🔬</div>
            <div style="font-size: 13px;">
                Configure the target parameters above and click <b style="color:#38BDF8;">Run Benchmark</b>.<br>
                Measures sequential time, parallel time, speedup, and numerical Pi accuracy.
            </div>
        </div>
        """, unsafe_allow_html=True)
