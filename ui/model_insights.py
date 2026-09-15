# ui/model_insights.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ui.utils import load_model, load_dataset, get_model_metrics

def render_model_insights():
    st.markdown("""
    <div class="page-header">
        <div class="title">🧠 Model Insights</div>
        <div class="subtitle">Evaluate Random Forest regression performance, feature importance, and generalization metrics.</div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    metrics = get_model_metrics()

    # Model Configuration & Empirical Evaluation Metrics
    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""<div class="card-box-header">📋 Random Forest Regressor Specifications</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="card-box" style="padding: 16px;">
            <div style="display: flex; flex-direction: column; gap: 10px; font-size: 13px;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 6px;">
                    <span style="color: #94A3B8;">Model Type</span>
                    <span style="font-weight: 600; color: #38BDF8;">RandomForestRegressor</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 6px;">
                    <span style="color: #94A3B8;">Target Variable</span>
                    <span style="font-weight: 600; color: #10B981;">Speedup (S = T1 / Tp)</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 6px;">
                    <span style="color: #94A3B8;">Input Features</span>
                    <span style="font-weight: 600; color: #818CF8;">N, threads, schedule, chunk</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 6px;">
                    <span style="color: #94A3B8;">Validation</span>
                    <span style="font-weight: 600; color: #F8FAFC;">GroupKFold CV + Holdout</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94A3B8;">Preprocessing</span>
                    <span style="font-weight: 600; color: #F8FAFC;">OneHotEncoder + Passthrough</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown("""<div class="card-box-header">🎯 Empirical Evaluation Metrics</div>""", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("R² Determination", f"{metrics['r2']:.4f}")
        m2.metric("Mean Abs Error (MAE)", f"{metrics['mae']:.4f}")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        m3, m4 = st.columns(2)
        m3.metric("Root Mean Sq (RMSE)", f"{metrics['rmse']:.4f}")
        m4.metric("Mean Abs % Error (MAPE)", f"{metrics['mape']:.2f}%")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Feature Importance Section
    col_feat, col_exp = st.columns([1.2, 1.0])

    with col_feat:
        st.markdown("""<div class="card-box-header">📊 Feature Importance Weights</div>""", unsafe_allow_html=True)
        if model is not None:
            try:
                rf_regressor = model.named_steps['regressor']
                preprocessor = model.named_steps['preprocessor']
                
                cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(['schedule']))
                num_names = ['N', 'threads', 'chunk']
                feature_names = cat_names + num_names
                importances = rf_regressor.feature_importances_

                feature_map = {
                    'Threads': 0.0,
                    'Workload (N)': 0.0,
                    'Schedule Policy': 0.0,
                    'Chunk Size': 0.0
                }
                for name, imp in zip(feature_names, importances):
                    if 'schedule' in name:
                        feature_map['Schedule Policy'] += imp
                    elif name == 'threads':
                        feature_map['Threads'] += imp
                    elif name == 'N':
                        feature_map['Workload (N)'] += imp
                    elif name == 'chunk':
                        feature_map['Chunk Size'] += imp

                fig, ax = plt.subplots(figsize=(6, 3.4))
                fig.patch.set_facecolor('#131B2E')
                ax.set_facecolor('#090D16')

                feat_series = pd.Series(feature_map).sort_values()
                feat_series.plot(kind='barh', ax=ax, color='#38BDF8', width=0.5)
                ax.set_title("Relative Feature Importance", color='#F8FAFC', fontsize=10, pad=8)
                ax.set_xlabel("Importance Weight", color='#94A3B8', fontsize=8)
                ax.tick_params(colors='#94A3B8', labelsize=8)
                ax.grid(axis='x', linestyle='--', alpha=0.15, color='#94A3B8')
                for spine in ax.spines.values(): spine.set_color('#1E293B')
                plt.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.info(f"Feature importance notice: {e}")
        else:
            st.info("Train Random Forest model via 'make train' to view feature importances.")

    with col_exp:
        st.markdown("""<div class="card-box-header">📖 Academic Model Explanation</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="card-box" style="border-left: 4px solid #38BDF8;">
            <div style="font-size: 13px; color: #F8FAFC; line-height: 1.6;">
                The Random Forest regression model learns relationships between workload characteristics ($N$), 
                OpenMP configuration parameters (thread counts, scheduling policies, chunk sizes), and empirical parallel speedup. 
                <br><br>
                <b>Technical Clarification:</b><br>
                The ML model does <i>not</i> directly control CPU kernel execution or OS thread dispatching. 
                Instead, it evaluates candidate configurations off-line and recommends high-performing parameter combinations, 
                which are subsequently verified through live C/OpenMP execution.
            </div>
        </div>
        """, unsafe_allow_html=True)
