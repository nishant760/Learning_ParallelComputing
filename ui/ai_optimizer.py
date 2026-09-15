# ui/ai_optimizer.py
import streamlit as st
import pandas as pd
import numpy as np

from ui.utils import (
    load_model,
    load_dataset,
    measure_median_time,
    save_experiment_record,
    ensure_binaries,
    SEQ_BIN,
    PAR_BIN,
)

# Sequential time empirically scales linearly with N above ~10M iterations.
# Derived from the benchmark dataset: ~0.878 ns/iteration.
SEQ_NS_PER_ITER = 0.878e-9   # seconds per iteration

def _estimate_seq_time(N, df):
    """Estimate sequential execution time for a given N.
    Uses dataset if N is in it, otherwise linear interpolation/extrapolation."""
    if df is not None and not df.empty:
        seq_rows = df[df['threads'] == 1].groupby('N')['median_time'].median()
        if N in seq_rows.index:
            return float(seq_rows[N])
        # linear interpolation/extrapolation using the per-iteration rate from dataset
        rates = seq_rows / seq_rows.index.astype(float)
        rate  = float(rates.mean())
        return N * rate
    return N * SEQ_NS_PER_ITER


def render_ai_optimizer():
    st.markdown("""
    <div class="page-header">
        <div class="title">🤖 AI-Assisted Scheduling</div>
        <div class="subtitle">
            Objective: Use the trained <b>Random Forest regression model</b> to predict
            <b>task burst time</b> and <b>expected speedup</b> for an unseen workload —
            without running the binary. Then validate by executing the actual C/OpenMP program
            and compare <b>predicted vs real</b> performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    df    = load_dataset()

    if model is None:
        st.error("Model not found at `model/speedup_model.pkl`. Run `make train` to train it.")
        return

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.markdown("""<div class="card-box-header">📥 Enter Workload Parameters</div>""", unsafe_allow_html=True)

    col_n, col_t, col_c, col_btn = st.columns([2.2, 1, 1, 1.2])

    with col_n:
        target_N = st.number_input(
            "Workload Size N (iterations)",
            min_value=1_000_000,
            max_value=500_000_000,
            value=50_000_000,
            step=5_000_000,
            format="%d",
        )

    with col_t:
        target_threads = st.selectbox(
            "Threads",
            options=[1, 2, 4, 8],
            index=2,
        )

    with col_c:
        target_chunk = st.selectbox(
            "Chunk Size",
            options=[100, 1000, 10000],
            index=1,
        )

    with col_btn:
        st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🤖 Predict", use_container_width=True, type="primary")

    # ── Run prediction ────────────────────────────────────────────────────────
    if predict_btn:
        schedules = ["static", "dynamic", "guided"]
        rows = [{"N": target_N, "threads": target_threads,
                 "schedule": s, "chunk": target_chunk} for s in schedules]
        cand_df = pd.DataFrame(rows)
        cand_df["predicted_speedup"] = model.predict(cand_df)

        t_seq_est = _estimate_seq_time(target_N, df)
        cand_df["est_seq_time"]  = round(t_seq_est, 6)
        cand_df["est_par_time"]  = (t_seq_est / cand_df["predicted_speedup"]).round(6)

        st.session_state["ai_pred"] = {
            "N":       target_N,
            "threads": target_threads,
            "chunk":   target_chunk,
            "df":      cand_df,
            "t_seq":   t_seq_est,
        }
        st.session_state.pop("ai_validation", None)

    # ── Show predictions ──────────────────────────────────────────────────────
    if "ai_pred" not in st.session_state:
        st.markdown("""
        <div class="card-box" style="text-align:center; padding: 40px 20px; color:#64748B;">
            <div style="font-size: 36px; margin-bottom: 8px;">🤖</div>
            <div style="font-size: 14px;">
                Fill in N, Threads, and Chunk Size above, then click
                <b style="color:#38BDF8;">Predict</b>.<br>
                The AI will instantly show predicted speedup and estimated time for all 3 schedulers.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    p      = st.session_state["ai_pred"]
    pred_df = p["df"]

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="card-box-header">📊 AI Predictions — N = {p['N']:,} · {p['threads']} Threads · Chunk {p['chunk']}</div>""",
        unsafe_allow_html=True,
    )

    # Estimated sequential time header metric
    st.markdown(
        f"""<div style="font-size:13px; color:#94A3B8; margin-bottom:14px;">
        Estimated sequential (1-thread) baseline: 
        <b style="color:#F8FAFC;">{p['t_seq']:.4f}s</b>
        &nbsp;—&nbsp; parallel times below are what the AI expects given this baseline.
        </div>""",
        unsafe_allow_html=True,
    )

    # One row of metrics per scheduler
    for _, row in pred_df.iterrows():
        sched  = row["schedule"].upper()
        sp     = row["predicted_speedup"]
        t_par  = row["est_par_time"]
        saving = (1 - t_par / p["t_seq"]) * 100

        color_map = {"STATIC": "#38BDF8", "DYNAMIC": "#818CF8", "GUIDED": "#10B981"}
        color = color_map.get(sched, "#F8FAFC")

        st.markdown(
            f"""<div style="font-size:12px; font-weight:700; color:{color};
                letter-spacing:.8px; text-transform:uppercase; margin-bottom:4px;">{sched}</div>""",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Speedup",    f"{sp:.2f}×")
        m2.metric("Est. Parallel Time",   f"{t_par:.4f}s")
        m3.metric("Est. Sequential Time", f"{p['t_seq']:.4f}s")
        m4.metric("Time Saving",          f"{saving:.1f}%")
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # Best pick
    best = pred_df.loc[pred_df["predicted_speedup"].idxmax()]
    st.markdown(
        f"""<div class="card-box" style="border-left:4px solid #10B981; margin-top:8px; padding:14px 20px;">
        <span style="font-size:11px; font-weight:700; color:#10B981; text-transform:uppercase; letter-spacing:1px;">
        ✦ AI Recommendation
        </span>
        <div style="font-size:13px; color:#F8FAFC; margin-top:6px;">
            Use <b style="color:#10B981;">{best['schedule'].upper()}</b> scheduling —
            predicted <b>{best['predicted_speedup']:.2f}×</b> speedup,
            estimated parallel time <b>{best['est_par_time']:.4f}s</b>
            vs sequential <b>{p['t_seq']:.4f}s</b>.
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Validate ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    col_v, _ = st.columns([2.2, 3])
    with col_v:
        validate_btn = st.button(
            "▶  Run C Binary & Compare Real vs Predicted",
            use_container_width=True,
            type="primary",
        )

    if validate_btn:
        ok, msg = ensure_binaries()
        if not ok:
            st.error(msg)
        else:
            schedules = ["static", "dynamic", "guided"]
            results   = []
            prog = st.progress(0, text="Measuring sequential baseline…")

            t_seq_real, _ = measure_median_time([SEQ_BIN, str(p["N"])])
            if not t_seq_real:
                st.error("Sequential binary failed. Run `make all`.")
                prog.empty()
            else:
                for i, sched in enumerate(schedules):
                    prog.progress((i + 1) / len(schedules),
                                  text=f"Running {sched.upper()}…")
                    t_par_real, _ = measure_median_time([
                        PAR_BIN, str(p["N"]),
                        str(p["threads"]), sched, str(p["chunk"])
                    ])
                    if t_par_real:
                        real_sp   = t_seq_real / t_par_real
                        pred_row  = pred_df[pred_df["schedule"] == sched].iloc[0]
                        pred_sp   = pred_row["predicted_speedup"]
                        err_pct   = abs(pred_sp - real_sp) / real_sp * 100
                        results.append({
                            "schedule":       sched,
                            "pred_speedup":   round(pred_sp, 4),
                            "real_speedup":   round(real_sp, 4),
                            "pred_par_time":  round(pred_row["est_par_time"], 6),
                            "real_par_time":  round(t_par_real, 6),
                            "error_pct":      round(err_pct, 2),
                        })
                        save_experiment_record({
                            "N":               p["N"],
                            "threads":         p["threads"],
                            "schedule":        sched,
                            "chunk":           p["chunk"],
                            "seq_time":        round(t_seq_real, 6),
                            "par_time":        round(t_par_real, 6),
                            "speedup":         round(real_sp, 2),
                            "efficiency":      round((real_sp / p["threads"]) * 100, 1),
                            "predicted_speedup": round(pred_sp, 2),
                        })

                prog.empty()
                if results:
                    st.session_state["ai_validation"] = {
                        "results":    results,
                        "t_seq_real": round(t_seq_real, 6),
                    }

    # ── Validation results ────────────────────────────────────────────────────
    if "ai_validation" in st.session_state:
        v = st.session_state["ai_validation"]

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class="card-box-header">✅ Real vs Predicted — Sequential baseline: {v['t_seq_real']}s</div>""",
            unsafe_allow_html=True,
        )

        color_map = {"static": "#38BDF8", "dynamic": "#818CF8", "guided": "#10B981"}
        for r in v["results"]:
            color = color_map.get(r["schedule"], "#F8FAFC")
            st.markdown(
                f"""<div style="font-size:12px; font-weight:700; color:{color};
                    letter-spacing:.8px; text-transform:uppercase; margin-bottom:4px;">
                    {r['schedule'].upper()}</div>""",
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("AI Predicted Speedup", f"{r['pred_speedup']:.2f}×")
            c2.metric("Real Speedup",          f"{r['real_speedup']:.2f}×")
            c3.metric("AI Est. Time",          f"{r['pred_par_time']:.4f}s")
            c4.metric("Real Time",             f"{r['real_par_time']:.4f}s")
            c5.metric("Prediction Error",      f"{r['error_pct']:.1f}%")
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # ── Reset ─────────────────────────────────────────────────────────────────
    if "ai_pred" in st.session_state:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset", use_container_width=False):
            for k in ("ai_pred", "ai_validation"):
                st.session_state.pop(k, None)
            st.rerun()
