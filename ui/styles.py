# ui/styles.py
import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
        /* Import Clean Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global Page Background & Font */
        .stApp {
            background-color: #090D16;
            color: #F1F5F9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Streamlit Main Block Padding */
        .main .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3.5rem;
            max-width: 1400px;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #94A3B8;
        }

        /* Header Title Banner */
        .page-header {
            margin-bottom: 28px;
            padding-bottom: 16px;
            border-bottom: 1px solid #1E293B;
        }
        .page-header .title {
            font-size: 28px;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.6px;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .page-header .subtitle {
            font-size: 14px;
            color: #94A3B8;
            margin-top: 6px;
            line-height: 1.5;
        }

        /* KPI Cards Styling */
        .kpi-card {
            background: #131B2E;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 18px 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transition: all 0.2s ease-in-out;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .kpi-card:hover {
            border-color: #38BDF8;
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(56, 189, 248, 0.1);
        }
        .kpi-title {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.9px;
            color: #64748B;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 24px;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.5px;
            line-height: 1.2;
        }
        .kpi-sub {
            font-size: 12px;
            color: #38BDF8;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Elevated Card Box */
        .card-box {
            background-color: #131B2E;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .card-box-header {
            font-size: 14px;
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Status Badge */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.12);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10B981;
            box-shadow: 0 0 10px #10B981;
        }

        /* Recommendation Box */
        .rec-box {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }
        .rec-header {
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #38BDF8;
            margin-bottom: 12px;
        }

        /* Pipeline Visualizer */
        .pipeline-flow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin: 24px 0;
            padding: 6px 0;
        }
        .pipeline-step {
            background: #131B2E;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            flex: 1;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }
        .pipeline-step .step-num {
            font-size: 10px;
            color: #38BDF8;
            font-weight: 800;
            letter-spacing: 0.5px;
        }
        .pipeline-step .step-name {
            font-size: 13px;
            font-weight: 600;
            color: #F8FAFC;
            margin-top: 4px;
        }
        .pipeline-arrow {
            color: #475569;
            font-size: 18px;
            font-weight: bold;
        }

        /* Disclaimer / Note Box */
        .hw-note {
            font-size: 12px;
            color: #F59E0B;
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-left: 4px solid #F59E0B;
            padding: 12px 16px;
            border-radius: 6px;
            margin-top: 20px;
            line-height: 1.5;
        }

        /* Streamlit Primary Button Overrides (Eliminate Neon Red) */
        button[data-testid="baseButton-primary"], 
        .stButton > button[data-testid="baseButton-primary"],
        button[type="primary"],
        div[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            background-color: #0284C7 !important;
            background-image: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #38BDF8 !important;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
        }
        button[data-testid="baseButton-primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover {
            background-color: #0369A1 !important;
            border-color: #0284C7 !important;
        }

        /* Streamlit Secondary Button Overrides */
        button[data-testid="baseButton-secondary"], 
        .stButton > button[data-testid="baseButton-secondary"] {
            background-color: #131B2E !important;
            color: #E2E8F0 !important;
            border: 1px solid #1E293B !important;
        }
        button[data-testid="baseButton-secondary"]:hover {
            border-color: #38BDF8 !important;
            color: #38BDF8 !important;
        }

        /* Form Control Improvements */
        .stSelectbox label, .stNumberInput label {
            font-size: 12px;
            color: #94A3B8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        /* Streamlit Metric Container Styling */
        div[data-testid="stMetric"] {
            background-color: #131B2E;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        div[data-testid="stMetricValue"] {
            font-size: 22px !important;
            font-weight: 800 !important;
            color: #F8FAFC !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 11px !important;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #94A3B8 !important;
            font-weight: 600 !important;
        }

        /* DataFrame Table Customization */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1E293B;
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)
