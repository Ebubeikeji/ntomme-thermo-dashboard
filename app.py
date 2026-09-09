import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

# ==============================================================================
# 0. PAGE CONFIGURATION & CUSTOM AESTHETICS
# ==============================================================================
st.set_page_config(
    page_title="Ntomme Subsea Calculator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS targeting Streamlit's stubborn default widgets
st.markdown("""
    <style>
    /* 1. Main Canvas */
    [data-testid="stAppViewContainer"] { background-color: #F5F0ED !important; }

    /* 1b. Streamlit renders icons (upload icon, sidebar collapse arrow) as text
       spans that rely on the Material Symbols font to turn e.g. "upload" into
       a glyph. Our font-family overrides below were breaking that font,
       which is why the raw icon names were showing up as literal text. */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* 2. Sidebar — pink, as requested. Darkened slightly from the original
       flat #F9D0D6 to #F7C2CE so navy text and the rose accent both read
       cleanly against it (the original was a touch too light for contrast). */
    [data-testid="stSidebar"] { background-color: #F7C2CE !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]),
    [data-testid="stSidebar"] label { color: #1A2E44 !important; }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
        color: #6A4A54 !important;
    }

    /* 3. Main canvas typography stays navy-on-cream */
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span:not([data-testid="stIconMaterial"]),
    [data-testid="stAppViewContainer"] label {
        color: #1A2E44 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* 4. File uploader — white card, rose dashed border and action button */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        border: 1.5px dashed #C43670 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        background-color: #C43670 !important;
        color: white !important;
    }

    /* 5. Multiselect dropdowns */
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #C43670 !important;
    }
    span[data-baseweb="tag"] {
        background-color: #C43670 !important;
    }
    span[data-baseweb="tag"] span {
        color: #FBD9E5 !important;
        font-weight: bold !important;
    }

    /* 6. Info alert box (main canvas) */
    [data-testid="stAlert"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DA9EA6 !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] * { color: #1A2E44 !important; }

    /* 7. Metric cards */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid #EBD8DC !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.03) !important;
    }

    /* 8. Buttons (main canvas) */
    .stButton>button, .stDownloadButton>button {
        background-color: #C43670 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #9E182B !important;
        color: white !important;
    }

    /* 9. About/contact — a collapsible expander instead of an always-open
       card, so it doesn't compete visually with the routing controls above it */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #C43670 !important;
        border-radius: 10px !important;
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
    mixed_cp = np.where(total_header_flow > 0.01, manifold_denominator / np.where(total_header_flow <= 0.01, 1.0, total_header_flow), (C_WATER + C_OIL + C_GAS) / 3.0)

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
# 5. FLOW PATH DIAGRAM (empty-state hero)
# ==============================================================================
def render_flow_diagram(h1_wells, h2_wells):
    h1_label = ", ".join(h1_wells) if h1_wells else "—"
    h2_label = ", ".join(h2_wells) if h2_wells else "—"
    # NOTE: rendered via components.html (not st.markdown) — Markdown treats
    # 4+ space indented lines as a code block, which was printing this as
    # literal text instead of parsing it as HTML/SVG.
    svg = f"""
<div style="background-color:#FFFFFF; border:1px solid #EBD8DC; border-radius:12px; padding:22px 28px 18px; font-family: 'Helvetica Neue', sans-serif; max-width:720px; margin:0 auto;">
<svg width="100%" viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="#8A97A6" stroke-width="1.5"/>
</marker>
</defs>
<text x="0" y="20" font-size="13" font-weight="600" fill="#4A5568">Flow path</text>

<rect x="40" y="50" width="140" height="48" rx="8" fill="#E1F0EC" stroke="#2F8F7C"/>
<text x="110" y="70" text-anchor="middle" font-size="13" fill="#0F5A4A" font-weight="600">Header 1</text>
<text x="110" y="88" text-anchor="middle" font-size="11" fill="#0F5A4A">{h1_label}</text>

<rect x="40" y="150" width="140" height="48" rx="8" fill="#E1F0EC" stroke="#2F8F7C"/>
<text x="110" y="170" text-anchor="middle" font-size="13" fill="#0F5A4A" font-weight="600">Header 2</text>
<text x="110" y="188" text-anchor="middle" font-size="11" fill="#0F5A4A">{h2_label}</text>

<line x1="180" y1="74" x2="220" y2="108" stroke="#8A97A6" marker-end="url(#arr)"/>
<line x1="180" y1="174" x2="220" y2="136" stroke="#8A97A6" marker-end="url(#arr)"/>

<rect x="220" y="96" width="140" height="52" rx="8" fill="#1A2E44"/>
<text x="290" y="118" text-anchor="middle" font-size="13" fill="#FFFFFF" font-weight="600">Manifold</text>
<text x="290" y="136" text-anchor="middle" font-size="11" fill="#C7D1DC">mixed temp</text>

<line x1="360" y1="122" x2="400" y2="122" stroke="#8A97A6" marker-end="url(#arr)"/>
<rect x="400" y="96" width="90" height="52" rx="8" fill="#1A2E44"/>
<text x="445" y="122" text-anchor="middle" font-size="13" fill="#FFFFFF" font-weight="600">PLET 1</text>

<line x1="490" y1="122" x2="530" y2="122" stroke="#8A97A6" marker-end="url(#arr)"/>
<rect x="530" y="96" width="90" height="52" rx="8" fill="#1A2E44"/>
<text x="575" y="122" text-anchor="middle" font-size="13" fill="#FFFFFF" font-weight="600">PLET 2</text>

<line x1="575" y1="148" x2="575" y2="180" stroke="#8A97A6" marker-end="url(#arr)"/>
<rect x="505" y="180" width="140" height="48" rx="8" fill="#C43670"/>
<text x="575" y="204" text-anchor="middle" font-size="13" fill="#FFFFFF" font-weight="600">Riser base</text>

<text x="340" y="248" text-anchor="middle" font-size="11" fill="#7A8794">Temperatures populate at each stage once a PI export is uploaded</text>
</svg>
</div>
"""
    return svg

# ==============================================================================
# 6. FRONTEND UI LAYOUT
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

    # -- ABOUT / CONTACT (collapsed by default) --
    with st.expander("👋 Built by Ebube"):
        st.markdown("""
            <p style="font-size:0.85rem; line-height:1.5; margin:0 0 10px;">
                Built to cross-check against APD readings so we can confirm accurate values.
            </p>
            <div style="display:flex; gap:16px;">
                <a href="mailto:ebubeikeji7@gmail.com" style="text-decoration:none; font-size:0.82rem; font-weight:600; color:#C43670;">✉️ Email</a>
                <a href="https://www.linkedin.com/in/ebube-ikeji/" target="_blank" style="text-decoration:none; font-size:0.82rem; font-weight:600; color:#1A2E44;">🔗 LinkedIn</a>
            </div>
        """, unsafe_allow_html=True)

if uploaded_file is None:
    components.html(render_flow_diagram(h1_selected, h2_selected), height=300)
else:
    with st.spinner('Calculating thermodynamic decay arrays...'):
        try:
            df = process_pi_data(uploaded_file)

            h1_config = [WELL_SPECS[w] for w in h1_selected]
            h2_config = [WELL_SPECS[w] for w in h2_selected]

            h1_preds = run_predictions(df, "Header1", h1_config)
            h2_preds = run_predictions(df, "Header2", h2_config)

            # -- FLOW DIAGRAM (kept visible after upload too, as context) --
            components.html(render_flow_diagram(h1_selected, h2_selected), height=300)

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