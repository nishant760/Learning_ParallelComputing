# ui/scheduling.py
import os
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ui.utils import (
    measure_median_time,
    save_experiment_record,
    ensure_binaries,
    SEQ_BIN,
    PAR_BIN
)

SCHEDULES   = ["static", "dynamic", "guided"]
CHUNK_SIZES = [100, 1000, 10000]
SCHED_COLORS = {
    "static":  "#38BDF8",
    "dynamic": "#818CF8",
    "guided":  "#10B981",
}


def _styled_bar(ax, df_pivot, ylabel, title):
    colors = [SCHED_COLORS.get(c, "#F8FAFC") for c in df_pivot.columns]
    df_pivot.plot(kind="bar", ax=ax, color=colors, width=0.62, edgecolor="none")
    ax.set_title(title, color="#F8FAFC", fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Chunk Size", color="#94A3B8", fontsize=8)
    ax.set_ylabel(ylabel, color="#94A3B8", fontsize=8)
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.14, color="#94A3B8")
    ax.legend(
        facecolor="#131B2E", edgecolor="#1E293B",
        labelcolor="#F8FAFC", fontsize=8, framealpha=0.9
    )
    plt.xticks(rotation=0)
    for spine in ax.spines.values():
        spine.set_color("#1E293B")
    plt.tight_layout()


def render_scheduling_analysis():
    os.environ["MPLCONFIGDIR"] = "/tmp"

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <div class="title">⚡ OpenMP Scheduling Analysis</div>
        <div class="subtitle">
            Objective: Compare <b>Static</b>, <b>Dynamic</b>, and <b>Guided</b> scheduling strategies for
            parallelising a numerical integration workload (π midpoint rule) using
            <code>#pragma omp parallel for</code>. Measure execution time, speedup (S = T₁/Tₚ),
            and parallel efficiency (E = S/P × 100%) across chunk sizes and thread counts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Algorithm explainer cards ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div class="card-box" style="border-top: 3px solid #38BDF8;">
        <div style="font-size: 14px; font-weight: 700; color: #38BDF8; letter-spacing:.5px;">STATIC</div>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px; line-height: 1.55;">
            Divides loop iterations into fixed-size chunks assigned round-robin at startup.
            Zero runtime overhead — best for uniform workloads.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("""
    <div class="card-box" style="border-top: 3px solid #818CF8;">
        <div style="font-size: 14px; font-weight: 700; color: #818CF8; letter-spacing:.5px;">DYNAMIC</div>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px; line-height: 1.55;">
            Idle threads fetch the next chunk from a shared queue at runtime.
            Adapts to load imbalance but incurs locking overhead.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown("""
    <div class="card-box" style="border-top: 3px solid #10B981;">
        <div style="font-size: 14px; font-weight: 700; color: #10B981; letter-spacing:.5px;">GUIDED</div>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px; line-height: 1.55;">
            Starts with large chunks that shrink exponentially to the minimum chunk size,
            balancing throughput and dynamic load levelling.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Control panel ────────────────────────────────────────────────────────
    st.markdown("""<div class="card-box-header">⚙️ Configure & Run Demo</div>""", unsafe_allow_html=True)

    col_n, col_t, col_c, col_btn = st.columns([2, 1, 1.4, 1.2])

    with col_n:
        selected_N = st.selectbox(
            "Workload Size (N — loop iterations)",
            options=[10_000_000, 20_000_000, 50_000_000, 80_000_000, 100_000_000],
            format_func=lambda x: f"{x // 1_000_000}M  ({x:,})",
            index=2,
        )

    with col_t:
        selected_threads = st.selectbox(
            "OpenMP Threads",
            options=[1, 2, 4, 8],
            index=2,
        )

    with col_c:
        selected_chunks = st.multiselect(
            "Chunk Sizes",
            options=[100, 1000, 10000],
            default=[100, 1000, 10000],
        )

    with col_btn:
        st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        run_demo = st.button(
            "▶  Run All Schedulers",
            use_container_width=True,
            type="primary",
        )

    chunk_label = " / ".join(str(c) for c in selected_chunks) if selected_chunks else "none selected"
    total_runs  = len(SCHEDULES) * len(selected_chunks) if selected_chunks else 0
    st.markdown(
        f"<div style='font-size:12px; color:#64748B; margin-top:4px; margin-bottom:4px;'>"
        f"Runs: Static · Dynamic · Guided  ×  Chunk sizes {chunk_label}  →  {total_runs} parallel + 1 sequential baseline"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Live execution ───────────────────────────────────────────────────────
    if run_demo:
        if not selected_chunks:
            st.warning("Please select at least one chunk size.")
            st.stop()

        ensure_ok, msg = ensure_binaries()
        if not ensure_ok:
            st.error(f"Build error: {msg}")
            return

        total_runs = len(SCHEDULES) * len(selected_chunks)
        progress_bar = st.progress(0, text="Measuring sequential baseline…")

        t_seq, _ = measure_median_time([SEQ_BIN, str(selected_N)])
        if not t_seq:
            st.error("Sequential binary failed to run. Check that binaries are built (`make all`).")
            progress_bar.empty()
            return

        records = []
        done = 0
        for sched in SCHEDULES:
            for chunk in selected_chunks:
                done += 1
                progress_bar.progress(
                    done / total_runs,
                    text=f"Running {sched.upper()} · chunk {chunk}  ({done}/{total_runs})…",
                )
                t_par, _ = measure_median_time(
                    [PAR_BIN, str(selected_N), str(selected_threads), sched, str(chunk)]
                )
                if t_seq and t_par:
                    speedup    = t_seq / t_par
                    efficiency = (speedup / selected_threads) * 100.0
                    records.append({
                        "schedule":        sched,
                        "chunk":           chunk,
                        "execution_time":  round(t_par, 6),
                        "speedup":         round(speedup, 4),
                        "efficiency":      round(efficiency, 1),
                    })
                    save_experiment_record({
                        "N":        selected_N,
                        "threads":  selected_threads,
                        "schedule": sched,
                        "chunk":    chunk,
                        "seq_time": round(t_seq, 6),
                        "par_time": round(t_par, 6),
                        "speedup":  round(speedup, 2),
                        "efficiency": round(efficiency, 1),
                    })

        progress_bar.empty()

        if records:
            st.session_state["sched_df"]      = pd.DataFrame(records)
            st.session_state["sched_N"]       = selected_N
            st.session_state["sched_threads"] = selected_threads
            st.session_state["sched_t_seq"]   = t_seq
        else:
            st.error("No results were collected — binary execution may have failed.")
            return

    # ── Results (shown only after a run) ─────────────────────────────────────
    if "sched_df" not in st.session_state or st.session_state["sched_df"].empty:
        st.markdown("""
        <div class="card-box" style="text-align:center; padding: 40px 20px; color:#64748B;">
            <div style="font-size: 36px; margin-bottom: 8px;">▶</div>
            <div style="font-size: 14px;">
                Configure N and Threads above, then click <b style="color:#38BDF8;">Run All Schedulers</b>.<br>
                Results and comparison graphs will appear here automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    df      = st.session_state["sched_df"].copy()
    run_N   = st.session_state.get("sched_N", selected_N)
    run_thr = st.session_state.get("sched_threads", selected_threads)
    t_seq   = st.session_state.get("sched_t_seq", None)

    sorted_df = df.sort_values("speedup", ascending=False).reset_index(drop=True)
    sorted_df.insert(0, "Rank", range(1, len(sorted_df) + 1))

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="card-box-header">⚡ Live Results — N = {run_N:,} · {run_thr} Threads</div>""",
        unsafe_allow_html=True,
    )

    # ── Ranked table ─────────────────────────────────────────────────────────
    display_df = sorted_df[["Rank", "schedule", "chunk", "execution_time", "speedup", "efficiency"]].copy()
    display_df.columns = ["Rank", "Schedule", "Chunk Size", "Exec Time (s)", "Speedup (×)", "Efficiency (%)"]

    st.dataframe(
        display_df.style.highlight_max(axis=0, subset=["Speedup (×)"], color="rgba(16, 185, 129, 0.22)"),
        use_container_width=True,
        height=230,
    )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Two side-by-side charts ───────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)

    pivot_speedup = sorted_df.pivot(index="chunk", columns="schedule", values="speedup")
    pivot_speedup = pivot_speedup[[c for c in SCHEDULES if c in pivot_speedup.columns]]

    pivot_time = sorted_df.pivot(index="chunk", columns="schedule", values="execution_time")
    pivot_time = pivot_time[[c for c in SCHEDULES if c in pivot_time.columns]]

    with col_g1:
        st.markdown("""<div class="card-box-header">📈 Speedup by Chunk Size</div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#131B2E")
        ax.set_facecolor("#090D16")
        _styled_bar(ax, pivot_speedup, "Speedup (×)", f"Schedule Speedup — N={run_N//1_000_000}M, {run_thr} Threads")
        # mark best bar
        best_sp = sorted_df.iloc[0]["speedup"]
        ax.axhline(best_sp, color="#F59E0B", linewidth=1, linestyle="--", alpha=0.6, label=f"Best: {best_sp:.2f}×")
        ax.legend(facecolor="#131B2E", edgecolor="#1E293B", labelcolor="#F8FAFC", fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_g2:
        st.markdown("""<div class="card-box-header">⏱ Execution Time by Chunk Size</div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#131B2E")
        ax.set_facecolor("#090D16")
        _styled_bar(ax, pivot_time, "Time (s)", f"Execution Time — N={run_N//1_000_000}M, {run_thr} Threads")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Insight panel ─────────────────────────────────────────────────────────
    best  = sorted_df.iloc[0]
    worst = sorted_df.iloc[-1]
    gap_pct = ((worst["speedup"] - best["speedup"]) / best["speedup"] * 100) if best["speedup"] else 0

    # Sequential baseline comparison
    seq_note = ""
    if t_seq:
        seq_note = (
            f"Sequential baseline: <b>{t_seq:.4f}s</b> → "
            f"parallel best cuts runtime by <b>{(1 - best['execution_time']/t_seq)*100:.1f}%</b>."
        )

    st.markdown(f"""
    <div class="card-box" style="border-left: 4px solid #10B981; margin-top: 6px;">
        <div style="font-size: 11px; font-weight: 700; color: #10B981; letter-spacing: 1px; text-transform:uppercase;">
            🏆 Demonstration Insight
        </div>
        <div style="font-size: 13px; color: #F8FAFC; margin-top: 8px; line-height: 1.6;">
            <b>Best:</b> <b style="color:#38BDF8;">{best['schedule'].upper()}</b> scheduling with
            chunk&nbsp;size&nbsp;<b>{int(best['chunk'])}</b> achieved
            <b style="color:#10B981;">{best['speedup']:.2f}× speedup</b>
            ({best['execution_time']:.4f}s).<br>
            <b>Worst:</b> <b style="color:#F87171;">{worst['schedule'].upper()}</b> with
            chunk&nbsp;size&nbsp;<b>{int(worst['chunk'])}</b> — {worst['speedup']:.2f}× speedup
            ({worst['execution_time']:.4f}s).<br>
            <span style="color:#94A3B8;">{seq_note}</span>
        </div>
        <div style="font-size: 12px; color: #F59E0B; margin-top: 8px; font-weight: 600;">
            ✦ Recommendation: use <b>{best['schedule'].upper()}</b> with chunk&nbsp;{int(best['chunk'])}
            for N&nbsp;=&nbsp;{run_N//1_000_000}M workloads on {run_thr}&nbsp;threads.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Reset button ──────────────────────────────────────────────────────────
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Clear Results & Run Again", use_container_width=False):
        for k in ("sched_df", "sched_N", "sched_threads", "sched_t_seq"):
            st.session_state.pop(k, None)
        st.rerun()
