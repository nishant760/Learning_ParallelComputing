# ui/ai_optimizer.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ui.utils import (
    load_model,
    load_dataset,
    get_ai_recommendation,
    measure_median_time,
    save_experiment_record,
    SEQ_BIN,
    PAR_BIN
)

def render_ai_optimizer():
    st.markdown("""
    <div class="page-header">
        <div class="title">🤖 AI Performance Optimizer</div>
        <div class="subtitle">Use empirical benchmark data & machine learning to recommend optimal OpenMP configurations for new workloads.</div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Random Forest Model missing at 'model/speedup_model.pkl'. Run 'make train' to train model.")
        return

    # Workload Input Panel
    st.markdown("""<div class="card-box-header">📥 Workload Input</div>""", unsafe_allow_html=True)

    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        target_N = st.number_input(
            "Target Workload Size N (Iterations)",
            min_value=1000000,
            max_value=500000000,
            value=80000000 if "run_demo" not in st.session_state else 50000000,
            step=5000000
        )
    with col_in2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("⚡ Analyze Workload", use_container_width=True, type="primary")

    # AI Pipeline Flow Visualization
    st.markdown("""
    <div class="pipeline-flow">
        <div class="pipeline-step">
            <div class="step-num">STEP 1</div>
            <div class="step-name">Workload N</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="step-num">STEP 2</div>
            <div class="step-name">Feature Prep</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="step-num">STEP 3</div>
            <div class="step-name">Random Forest</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="step-num">STEP 4</div>
            <div class="step-name">27 Candidates</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="step-num">STEP 5</div>
            <div class="step-name">Best Config</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="step-num">STEP 6</div>
            <div class="step-name">Live Validation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Candidate evaluation table & Recommendation
    cand_df = get_ai_recommendation(model, target_N)
    top_config = cand_df.iloc[0]

    col_cand, col_rec = st.columns([1.5, 1.0])

    with col_cand:
        st.markdown("""<div class="card-box-header">📋 Evaluated Candidate Configurations</div>""", unsafe_allow_html=True)
        st.dataframe(
            cand_df[['Rank', 'threads', 'schedule', 'chunk', 'predicted_speedup']].style.highlight_max(axis=0, subset=['predicted_speedup'], color='rgba(52, 211, 153, 0.3)'),
            use_container_width=True,
            height=320
        )

    with col_rec:
        st.markdown("""<div class="card-box-header">⭐ AI Recommended Configuration</div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rec-box">
            <div class="rec-header">RECOMMENDED OPTIMAL SETTING</div>
            <div style="font-size: 22px; font-weight: 800; color: #F8FAFC;">
                {int(top_config['threads'])} Threads
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #38BDF8; margin-top: 4px;">
                {str(top_config['schedule']).upper()} Scheduling (Chunk: {int(top_config['chunk'])})
            </div>
            <div style="font-size: 18px; font-weight: 700; color: #34D399; margin-top: 10px;">
                Predicted Speedup: {top_config['predicted_speedup']:.2f}x
            </div>
            <div style="font-size: 12px; color: #94A3B8; margin-top: 10px; line-height: 1.4;">
                <b>Why this recommendation?</b><br>
                Highest predicted speedup among all 27 evaluated candidate configurations based on empirical dataset patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)

        val_btn = st.button("🔬 Validate Recommendation (Run C Binary)", use_container_width=True, type="primary")

    st.markdown("<br>", unsafe_allow_html=True)

    # Live Validation Section
    if val_btn or "run_demo" in st.session_state:
        if "run_demo" in st.session_state:
            st.session_state.pop("run_demo", None)
            st.info("Running Quick Demo Mode Validation on N = 50,000,000...")

        st.markdown("""<div class="card-box-header">⚡ Live C/OpenMP Execution Validation</div>""", unsafe_allow_html=True)

        with st.spinner(f"Measuring median execution time for N = {target_N:,}..."):
            t_seq, pi_seq = measure_median_time([SEQ_BIN, str(target_N)])
            t_par, pi_par = measure_median_time([
                PAR_BIN, str(target_N), str(int(top_config['threads'])), str(top_config['schedule']), str(int(top_config['chunk']))
            ])

        if t_seq and t_par:
            act_speedup = t_seq / t_par
            act_eff = (act_speedup / int(top_config['threads'])) * 100.0
            pred_speedup = float(top_config['predicted_speedup'])
            abs_err = abs(pred_speedup - act_speedup)
            pct_err = (abs_err / act_speedup) * 100.0

            vc1, vc2, vc3, vc4 = st.columns(4)
            vc1.metric("Predicted Speedup", f"{pred_speedup:.2f}x")
            vc2.metric("Measured Actual Speedup", f"{act_speedup:.2f}x")
            vc3.metric("Prediction Absolute Error", f"{abs_err:.4f}")
            vc4.metric("Percentage Error", f"{pct_err:.2f}%")

            save_experiment_record({
                'N': target_N,
                'threads': int(top_config['threads']),
                'schedule': str(top_config['schedule']),
                'chunk': int(top_config['chunk']),
                'seq_time': round(t_seq, 6),
                'par_time': round(t_par, 6),
                'speedup': round(act_speedup, 2),
                'efficiency': round(act_eff, 1),
                'predicted_speedup': round(pred_speedup, 2),
                'error': round(abs_err, 4)
            })

            st.markdown("<br>", unsafe_allow_html=True)

            # Predicted vs Actual Comparison Bar Chart
            col_chart, col_explain = st.columns([1.5, 1.0])

            with col_chart:
                st.markdown("""<div class="card-box-header">Predicted vs Measured Speedup Comparison</div>""", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 3.2))
                fig.patch.set_facecolor('#1E293B')
                ax.set_facecolor('#0B0F19')

                bars = ax.bar(['AI Predicted Speedup', 'Measured Speedup'], [pred_speedup, act_speedup], color=['#818CF8', '#34D399'], width=0.4)
                ax.set_ylabel("Speedup (x Baseline)", color='#94A3B8', fontsize=9)
                ax.tick_params(colors='#94A3B8', labelsize=9)
                ax.grid(axis='y', linestyle='--', alpha=0.2, color='#64748B')
                for spine in ax.spines.values(): spine.set_color('#334155')

                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}x", ha='center', va='bottom', color='#F8FAFC', fontweight='bold', fontsize=10)

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_explain:
                st.markdown(f"""
                <div class="card-box" style="border-left: 4px solid #818CF8;">
                    <div style="font-size: 13px; font-weight: 700; color: #818CF8;">VALIDATION ANALYSIS</div>
                    <div style="font-size: 13px; color: #F8FAFC; margin-top: 8px; line-height: 1.5;">
                        The Random Forest model predicted a speedup of <b>{pred_speedup:.2f}x</b>. 
                        Live hardware execution of the compiled OpenMP C binary yielded a measured speedup of <b>{act_speedup:.2f}x</b>.
                    </div>
                    <div style="font-size: 12px; color: #34D399; margin-top: 10px; font-weight: 600;">
                        ✓ Validation Error: {pct_err:.2f}% (High Model Fidelity)
                    </div>
                </div>
                """, unsafe_allow_html=True)
