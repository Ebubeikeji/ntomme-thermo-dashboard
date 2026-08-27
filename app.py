import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

# ==============================================================================
# 0. PAGE CONFIGURATION & AESTHETICS
# ==============================================================================
st.set_page_config(
    page_title="Ntomme Virtual Sensor Controller",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimalist CSS to match the "Sanctuary Science" aesthetic
st.markdown("""
    <style>
    .main {background-color: #ffffff;}
    h1, h2, h3 {color: #1a2e44; font-family: 'Helvetica Neue', sans-serif;}
    .stMetric {background-color: #f8fafc; border-radius: 8px; padding: 15px; border: 1px solid #e2e8f0;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CONSTANTS & PI TAG MAPPING
# ==============================================================================
T_AMBIENT = 4.0      
RHO_OIL = 850.0      
RHO_WATER = 1030.0   
RHO_GAS = 0.8        
C_WATER = 4200.0     
C_OIL = 2000.0       
C_GAS = 2200.0       
K_JUMPER = 89.40    
K_FLOWLINE = 1.2865  

NTOMME_FLOWPATHS = {'man_to_plet1': 152.108, 'plet1_to_plet2': 7217.0, 'plet2_to_rb': 150.471}
PLET_PENALTY_LENGTH = 75.0

PI_TAG_MAPPING = {
    'TI-0521203A.PV': 'W1_Temp', 'FI-0521202.PV': 'W1_OilFlow_bopd', 'FI-0521206.PV': 'W1_WaterFlow_bwpd', 'FI-0521204.PV': 'W1_GasFlow_mmscfd',
    'TI-0512101A.PV': 'W3_Temp', 'FI-0512102.PV': 'W3_OilFlow_bopd', 'FI-0512106.PV': 'W3_WaterFlow_bwpd', 'FI-0512104.PV': 'W3_GasFlow_mmscfd',
    'TI-0513201B.PV': 'W5_Temp', 'FI-0513202.PV': 'W5_OilFlow_bopd', 'FI-0513206.PV': 'W5_WaterFlow_bwpd', 'FI-0513204.PV': 'W5_GasFlow_mmscfd',
    'TI-0521403A.PV': 'W9_Temp', 'FI-0521402.PV': 'W9_OilFlow_bopd', 'FI-0521406.PV': 'W9_WaterFlow_bwpd', 'FI-0521404.PV': 'W9_GasFlow_mmscfd',
}

# ==============================================================================
# 2. CORE THERMODYNAMIC FUNCTIONS
# ==============================================================================
def convert_to_mass_flow(bopd, bwpd, mmscfd):
    m_oil = (bopd * 0.158987 / 86400.0) * RHO_OIL
    m_water = (bwpd * 0.158987 / 86400.0) * RHO_WATER
    m_gas = (mmscfd * 28316.8 / 86400.0) * RHO_GAS
    return m_water, m_oil, m_gas

def calculate_mixture_cp(m_water, m_oil, m_gas):
    m_total = m_water + m_oil + m_gas
    weighted_cp = ((m_water * C_WATER) + (m_oil * C_OIL) + (m_gas * C_GAS)) / np.where(m_total == 0, 1.0, m_total)
    return np.where(m_total <= 0.001, (C_WATER + C_OIL + C_GAS) / 3.0, weighted_cp)

def calculate_thermal_decay(t_in, mass_flow, cp_mix, length, k_constant):
    safe_flow = np.where(mass_flow <= 0.01, 1.0, mass_flow)
    exponent = - (k_constant * length) / (safe_flow * cp_mix)
    return np.where(mass_flow <= 0.01, T_AMBIENT, T_AMBIENT + (t_in - T_AMBIENT) * np.exp(exponent))

# ==============================================================================
# 3. DATA INGESTION & PROCESSING (Cached for speed)
# ==============================================================================
@st.cache_data
def process_pi_data(uploaded_file):
    # 1. Dynamically find the header row by looking for '.PV'
    raw_df = pd.read_excel(uploaded_file, header=None)
    header_idx = 0
    for idx, row in raw_df.head(20).iterrows():
        if any(isinstance(val, str) and '.PV' in val for val in row.values):
            header_idx = idx
            break
            
    # 2. Read the dataframe using the correct header
    df = pd.read_excel(uploaded_file, skiprows=header_idx)
    
    # 3. Rename "Unnamed" timestamp column and set as index
    time_col = [col for col in df.columns if 'unnamed' in str(col).lower() or 'time' in str(col).lower()][0]
    df.rename(columns={time_col: 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    # 4. Map PI Tags to internal variables
    df.rename(columns=PI_TAG_MAPPING, inplace=True)
    
    # 5. Handle Units & Missing Data
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'GasFlow' in col:
            df[col] = df[col] * 0.000848  
        elif 'OilFlow' in col or 'WaterFlow' in col:
            df[col] = df[col] * 150.96    
    return df.ffill().fillna(0.0)

@st.cache_data
def run_predictions(df, header_name, wells_config):
    results_df = pd.DataFrame(index=df.index)
    total_header_flow, manifold_numerator, manifold_denominator = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))

    for well in wells_config:
        name = well['name']
        
        # Ensure columns exist, fill with 0 if missing from PI export
        t_xt = df.get(f'{name}_Temp', np.zeros(len(df)))
        bwpd = df.get(f'{name}_WaterFlow_bwpd', np.zeros(len(df)))
        bopd = df.get(f'{name}_OilFlow_bopd', np.zeros(len(df)))
        mmscfd = df.get(f'{name}_GasFlow_mmscfd', np.zeros(len(df)))

        m_water, m_oil, m_gas = convert_to_mass_flow(bopd, bwpd, mmscfd)
        m_total = m_water + m_oil + m_gas
        cp = calculate_mixture_cp(m_water, m_oil, m_gas)

        t_arrival = calculate_thermal_decay(t_xt, m_total, cp, well['l_jumper'], K_JUMPER)
        
        manifold_numerator += (m_total * cp) * t_arrival
        manifold_denominator += (m_total * cp)
        total_header_flow += m_total

    safe_denom = np.where(manifold_denominator <= 0.01, 1.0, manifold_denominator)
    t_header_mixed = np.where(manifold_denominator > 0.01, manifold_numerator / safe_denom, T_AMBIENT)
    mixed_cp = np.where(total_header_flow > 0.01, manifold_denominator / np.where(total_header_flow <= 0.01, 1.0, total_header_flow), (C_WATER + C_OIL + C_GAS)/3.0)

    results_df[f'{header_name}_Temp'] = t_header_mixed
    t_plet1 = calculate_thermal_decay(t_header_mixed, total_header_flow, mixed_cp, NTOMME_FLOWPATHS['man_to_plet1'] + PLET_PENALTY_LENGTH, K_FLOWLINE)
    t_plet2 = calculate_thermal_decay(t_plet1, total_header_flow, mixed_cp, NTOMME_FLOWPATHS['plet1_to_plet2'] + PLET_PENALTY_LENGTH, K_FLOWLINE)
    results_df[f'{header_name}_Riser_Base_Temp'] = calculate_thermal_decay(t_plet2, total_header_flow, mixed_cp, NTOMME_FLOWPATHS['plet2_to_rb'], K_FLOWLINE)

    results_df.loc[total_header_flow < 0.1, [f'{header_name}_Temp', f'{header_name}_Riser_Base_Temp']] = np.nan
    return results_df

# ==============================================================================
# 4. PLOTTING FUNCTION
# ==============================================================================
def create_styled_plot(df, temp_col, riser_col, title, line1_color, line2_color, y_max):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    ax.plot(df.index, df[temp_col], label='Manifold Temp', color=line1_color, linewidth=2.0)
    ax.plot(df.index, df[riser_col], label='Riser Base Temp', color=line2_color, linewidth=2.5, linestyle=':')
    
    # Force fixed y-axis from 0 to match standard APD report scaling
    ax.set_ylim(0, y_max)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("DATE", fontsize=10, fontweight='bold', color='#333333')
    ax.set_ylabel("TEMPERATURE (°C)", fontsize=10, fontweight='bold', color='#333333')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, which='major', axis='both', color='#e5e5e5')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
    fig.subplots_adjust(bottom=0.2)
    return fig

# ==============================================================================
# 5. FRONTEND UI LAYOUT
# ==============================================================================
st.title("Ntomme Virtual Sensor Controller")
st.markdown("Subsea Thermodynamic Decay Predictor")
st.divider()

with st.sidebar:
    st.header("Data Ingestion")
    st.markdown("""
    **Instructions:**
    1. Select required tags in PI Datalink.
    2. Set interval to **1 hour**.
    3. Export as `.xlsx` and upload below.
    """)
    uploaded_file = st.file_uploader("Upload PI Vision Excel File", type=['xlsx'])

if uploaded_file is None:
    st.info("👈 Please upload a weekly PI Vision Excel file in the sidebar to begin processing.")
else:
    with st.spinner('Crunching thermodynamic decay arrays...'):
        try:
            df = process_pi_data(uploaded_file)
            
            h1_preds = run_predictions(df, "Header1", [{'name': 'W3', 'l_jumper': 20.313}, {'name': 'W5', 'l_jumper': 21.034}])
            h2_preds = run_predictions(df, "Header2", [{'name': 'W1', 'l_jumper': 21.034}, {'name': 'W9', 'l_jumper': 20.313}])
            
            # -- DISPLAY METRIC CARDS --
            st.subheader("Latest System Status")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Header 1 (Manifold 4)", f"{h1_preds['Header1_Temp'].iloc[-1]:.1f} °C")
            col2.metric("Riser 21 Base", f"{h1_preds['Header1_Riser_Base_Temp'].iloc[-1]:.1f} °C")
            col3.metric("Header 2 (Manifold 4)", f"{h2_preds['Header2_Temp'].iloc[-1]:.1f} °C")
            col4.metric("Riser 20 Base", f"{h2_preds['Header2_Riser_Base_Temp'].iloc[-1]:.1f} °C")
            
            st.divider()
            
            
            # -- DISPLAY GRAPHS --
            st.subheader("Thermal Trend Analysis")
            fig1 = create_styled_plot(h1_preds, 'Header1_Temp', 'Header1_Riser_Base_Temp', "Header 1 to Riser 21 Flowline", '#f39200', '#1a2e44', 100)
            st.pyplot(fig1)
            
            fig2 = create_styled_plot(h2_preds, 'Header2_Temp', 'Header2_Riser_Base_Temp', "Header 2 to Riser 20 Flowline", '#0ea5e9', '#0369a1', 80)
            st.pyplot(fig2)
            
            # -- DOWNLOAD BUTTONS --
            st.markdown("### Export Presentation-Ready Graphs")
            col_a, col_b = st.columns(2)
            
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
            col_a.download_button(label="Download Riser 21 Graph (PNG)", data=buf1.getvalue(), file_name="riser21_predictions.png", mime="image/png")
            
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
            col_b.download_button(label="Download Riser 20 Graph (PNG)", data=buf2.getvalue(), file_name="riser20_predictions.png", mime="image/png")

        except Exception as e:
            st.error(f"Error processing file. Please ensure tags are correct. Details: {e}")