# ui/system_info.py
import streamlit as st
from ui.utils import get_system_info

def render_system_info():
    st.markdown("""
    <div class="page-header">
        <div class="title">💻 System & Hardware Information</div>
        <div class="subtitle">Live hardware environment details, OpenMP toolchain availability, and execution context.</div>
    </div>
    """, unsafe_allow_html=True)

    sys_info = get_system_info()

    # Hardware & Environment Cards Row
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Logical CPU Cores", sys_info['logical_cores'])
    col_s2.metric("Max OpenMP Threads", sys_info['max_openmp_threads'])
    col_s3.metric("Python Version", sys_info['python_version'])
    col_s4.metric("OpenMP Engine", "Connected")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed System Information Table
    st.markdown("""<div class="card-box-header">🖥️ Benchmark Environment Details</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card-box">
        <table style="width: 100%; font-size: 14px; color: #F8FAFC; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #334155; height: 40px;">
                <td style="padding: 8px; color: #94A3B8; font-weight: 500; width: 250px;">Host Processor (CPU)</td>
                <td style="padding: 8px; font-weight: 600; color: #38BDF8;">{sys_info['cpu_name']}</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155; height: 40px;">
                <td style="padding: 8px; color: #94A3B8; font-weight: 500;">Logical Core Count</td>
                <td style="padding: 8px; font-weight: 600; color: #F8FAFC;">{sys_info['logical_cores']} Cores</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155; height: 40px;">
                <td style="padding: 8px; color: #94A3B8; font-weight: 500;">OpenMP Thread Pool Capacity</td>
                <td style="padding: 8px; font-weight: 600; color: #34D399;">1, 2, 4, 8 Threads Supported</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155; height: 40px;">
                <td style="padding: 8px; color: #94A3B8; font-weight: 500;">Operating System & Kernel</td>
                <td style="padding: 8px; font-weight: 600; color: #F8FAFC;">{sys_info['os']}</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155; height: 40px;">
                <td style="padding: 8px; color: #94A3B8; font-weight: 500;">C Compiler Toolchain</td>
                <td style="padding: 8px; font-weight: 600; color: #818CF8;">{sys_info['compiler']}</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155; height: 40px;">
                <td style="padding: 8px; color: #94A3B8; font-weight: 500;">OpenMP API Version</td>
                <td style="padding: 8px; font-weight: 600; color: #34D399;">OpenMP 4.5 / 5.0 (omp.h)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hw-note">
        ⚠️ Hardware Dependency Notice: Parallel speedup, execution time, and core efficiency are inherently hardware-dependent. 
        Benchmark metrics displayed throughout this platform reflect the physical hardware capabilities and OS process scheduling of the current host environment.
    </div>
    """, unsafe_allow_html=True)
