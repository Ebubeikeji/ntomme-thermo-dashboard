import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

# ==============================================================================
# 0. PAGE CONFIGURATION & DUSTY ROSE / NAVY AESTHETICS
# ==============================================================================
st.set_page_config(
    page_title="Ntomme Virtual Sensor Controller",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a soft pink + professional subsea navy palette
st.markdown("""
    <style>
    /* Force Main Canvas Background (Overrides Streamlit Dark Mode) */
    [data-testid="stAppViewContainer"] { 
        background-color: #F8ECEE !important; 
    }
    
    /* Force Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #E2C2C9 !important;
    }
    
    /* Typography & Headers - Force Navy Blue Everywhere */
    h1, h2, h3, h4, p, span, label, div { 
        color: #1A2E44 !important; 
        font-family: 'Helvetica Neue', sans-serif; 
    }

    /* Metric Cards */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important; 
        border-radius: 12px !important; 
        padding: 16px !important; 
        border: 1px solid #DDAFC0 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }
    
    /* Buttons */
    .stButton>button, .stDownloadButton>button {
        background-color: #C87A8F !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #B2657A !important;
        color: white !important;
    }
    
    /* Info Box (File upload warning) */
    .stAlert {
        background-color: #FFFFFF !important;
        border: 1px solid #C87A8F !important;
    }

    /* Connect Box */
    .connect-card {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 18px !important;
        border: 1px solid #C87A8F !important;
        margin-top: 20px !important;
    }
    
    /* Fix specific text overrides in the connect card */
    .connect-card h4, .connect-card p {
        color: #1A2E44 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CONSTANTS, WELL METADATA & PI TAG MAPPING
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

WELL_SPECS = {
    'W1': {'name': 'W1', 'l_jumper': 21.034},
    'W3': {'name': 'W3', 'l_jumper': 20.313},
    'W5': {'name': 'W5', 'l_jumper': 21.034},
    'W9': {'name': 'W9', 'l_jumper': 20.313},
}

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
# 3. DATA INGESTION & PROCESSING
# ==============================================================================
@st.cache_data
def process_pi_data(uploaded_file):
    raw_df = pd.read_excel(uploaded_file, header=None)
    header_idx = 0
    for idx, row in raw_df.head(20).iterrows():
        if any(isinstance(val, str) and '.PV' in val for val in row.values):
            header_idx = idx
            break
            
    df = pd.read_excel(uploaded_file, skiprows=header_idx)
    time_col = [col for col in df.columns if 'unnamed' in str(col).lower() or 'time' in str(col).lower()][0]
    df.rename(columns={time_col: 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    df.rename(columns=PI_TAG_MAPPING, inplace=True)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'GasFlow' in col:
            df[col] = df[col] * 0.000848  
        elif 'OilFlow' in col or 'WaterFlow' in col:
            df[col] = df[col] * 150.96    
    return df.ffill().fillna(0.0)

def run_predictions(df, header_name, wells_config):
    results_df = pd.DataFrame(index=df.index)
    total_header_flow, manifold_numerator, manifold_denominator = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))

    for well in wells_config:
        name = well['name']
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
    
    ax.plot(df.index, df[temp_col], label='Manifold Temp', color=line1_color, linewidth=2.2)
    ax.plot(df.index, df[riser_col], label='Riser Base Temp', color=line2_color, linewidth=2.5, linestyle=':')
    
    ax.set_ylim(0, y_max)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#1A2E44', pad=15)
    ax.set_xlabel("DATE", fontsize=9, fontweight='bold', color='#4A5568')
    ax.set_ylabel("TEMPERATURE (°C)", fontsize=9, fontweight='bold', color='#4A5568')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, which='major', axis='both', color='#EDF2F7')
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
    uploaded_file = st.file_uploader("Upload Weekly PI Vision Excel File", type=['xlsx'])
    
    st.divider()
    st.header("Manifold Routing")
    st.caption("Reconfigure active wells per header whenever subsea alignments change.")
    
    h1_selected = st.multiselect("Header 1 Active Wells", options=list(WELL_SPECS.keys()), default=['W3', 'W5'])
    h2_selected = st.multiselect("Header 2 Active Wells", options=list(WELL_SPECS.keys()), default=['W1', 'W9'])

    # -- CREATOR & CONTACT CARD --
    st.markdown("""
        <div class="connect-card">
            <h4 style="margin-top:0; color:#1A2E44;">👋 Built by Ebube</h4>
            <p style="font-size: 0.88rem; line-height: 1.4; color: #4A5568;">
                I built this controller to serve as another data set just like we have the APD readings, we can compare both to ensure we have accurate readings!
            </p>
            <p style="font-size: 0.88rem; font-weight: 500; color: #1A2E44; margin-bottom: 8px;">
                Have any feedback or want to connect? Send me an email! I'd love to hear from you:
            </p>
            <a href="mailto:ebubeikeji7@gmail.com" style="text-decoration:none;">
                <p style="font-size: 0.85rem; font-weight: bold; color: #C87A8F; margin-bottom: 12px;">
                    ✉️ ebubeikeji7@gmail.com
                </p>
            </a>
            <a href="www.linkedin.com/in/ebube-ikeji" target="_blank" style="text-decoration:none;">
                <p style="font-size: 0.85rem; font-weight: bold; color: #1A2E44; margin:0;">
                    🔗 Connect on LinkedIn
                </p>
            </a>
        </div>
    """, unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👈 Please upload a weekly PI Vision Excel file in the sidebar to run thermal predictions.")
else:
    with st.spinner('Calculating thermodynamic decay arrays...'):
        try:
            df = process_pi_data(uploaded_file)
            
            h1_config = [WELL_SPECS[w] for w in h1_selected]
            h2_config = [WELL_SPECS[w] for w in h2_selected]
            
            h1_preds = run_predictions(df, "Header1", h1_config)
            h2_preds = run_predictions(df, "Header2", h2_config)
            
            # -- DISPLAY METRIC CARDS --
            st.subheader("Latest System Status")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Header 1 ({', '.join(h1_selected)})", f"{h1_preds['Header1_Temp'].iloc[-1]:.1f} °C")
            col2.metric("Riser 21 Base", f"{h1_preds['Header1_Riser_Base_Temp'].iloc[-1]:.1f} °C")
            col3.metric(f"Header 2 ({', '.join(h2_selected)})", f"{h2_preds['Header2_Temp'].iloc[-1]:.1f} °C")
            col4.metric("Riser 20 Base", f"{h2_preds['Header2_Riser_Base_Temp'].iloc[-1]:.1f} °C")
            
            st.divider()
            
            # -- DISPLAY GRAPHS --
            st.subheader("Thermal Trend Analysis")
            # Dusty Rose for Manifold, Subsea Navy for Riser Base
            fig1 = create_styled_plot(h1_preds, 'Header1_Temp', 'Header1_Riser_Base_Temp', f"Header 1 ({', '.join(h1_selected)}) to Riser 21 Flowline", '#C87A8F', '#1A2E44', 100)
            st.pyplot(fig1)
            
            fig2 = create_styled_plot(h2_preds, 'Header2_Temp', 'Header2_Riser_Base_Temp', f"Header 2 ({', '.join(h2_selected)}) to Riser 20 Flowline", '#D87093', '#1A2E44', 80)
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
            st.error(f"Error processing file. Please ensure tag headers match expected PI exports. Details: {e}")