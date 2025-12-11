"""
BTC Options Analysis - Main Streamlit Dashboard
Professional cryptocurrency options analytics with Greeks, IV, and Gamma Exposure
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors import OptionsDataFetcher
from src.models.greeks import EnhancedGreeksCalculator, PricingModel
from src.analytics.gamma_exposure import GammaExposureAnalyzer
from config.settings import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
Config.ensure_directories()

# Page configuration
st.set_page_config(
    page_title="BTC Options Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit',
        'Report a bug': 'https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit/issues',
        'About': '# BTC Options Analytics Dashboard\nProfessional options analysis for crypto markets'
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 Cryptocurrency Options Analytics</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")

    # Currency selection
    st.subheader("Asset Selection")
    currencies = st.multiselect(
        "Select Cryptocurrencies",
        options=Config.SUPPORTED_CURRENCIES,
        default=["BTC"],
        help="Choose one or more cryptocurrencies to analyze"
    )

    # Date range
    st.subheader("📅 Date Range")
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=7),
            max_value=date.today(),
            help="Start date for data collection"
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            max_value=date.today(),
            help="End date for data collection"
        )

    # Validate date range
    if start_date > end_date:
        st.error("❌ Start date must be before end date!")

    date_range_days = (end_date - start_date).days
    if date_range_days > Config.MAX_DATE_RANGE_DAYS:
        st.warning(f"⚠️ Date range limited to {Config.MAX_DATE_RANGE_DAYS} days")
        end_date = start_date + timedelta(days=Config.MAX_DATE_RANGE_DAYS)

    # Analytics settings
    st.subheader("🔧 Analytics Settings")

    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=Config.DEFAULT_RISK_FREE_RATE * 100,
        step=0.1,
        help="Annual risk-free interest rate (typically 0 for crypto)"
    ) / 100

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        use_cache = st.checkbox("Use Cached Data", value=True, help="Use cached data if available")
        max_workers = st.slider("Parallel Workers", 1, 10, Config.MAX_WORKERS, help="Number of parallel workers for data fetching")

        if st.button("🗑️ Clear Cache"):
            if 'data_fetcher' in st.session_state:
                st.session_state.data_fetcher.clear_cache()
            st.success("Cache cleared!")

    # Analysis button
    st.markdown("---")
    run_analysis = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        help="Fetch data and run complete analysis"
    )

    # Info
    st.markdown("---")
    st.info("""
    **📌 Quick Guide:**
    1. Select currencies
    2. Choose date range
    3. Click 'Run Analysis'
    4. Explore pages via sidebar
    """)

# Main content area
if not run_analysis and 'analysis_complete' not in st.session_state:
    # Welcome screen
    st.markdown("""
    ### Welcome to BTC Options Analytics Dashboard 👋

    This professional dashboard provides comprehensive analysis of cryptocurrency options from Deribit exchange.

    #### 🎯 Key Features:

    **📊 Market Overview**
    - Real-time options data
    - Call/Put ratio analysis
    - Volume metrics
    - Most active strikes

    **📈 Volatility Analysis**
    - 3D Volatility surfaces
    - IV skew by maturity
    - Weighted IV time series
    - Historical vs Implied volatility

    **🎯 Greeks Analysis**
    - Delta, Gamma, Vega, Theta surfaces
    - Greeks vs moneyness
    - Greeks vs IV
    - Portfolio risk metrics

    **⚡ Gamma Exposure (GEX)**
    - Gamma exposure by strike
    - **Gamma squeeze detection**
    - Net GEX profiles
    - Market maker positioning

    #### 🚀 Getting Started:
    1. Configure your analysis in the sidebar
    2. Click the **"Run Analysis"** button
    3. Navigate between pages using the sidebar menu
    4. All charts are interactive - hover for details!

    ---
    **📖 Data Source:** [Deribit Options API](https://www.deribit.com)
    """)

    # Display some info boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-box"><b>📊 Real-time Data</b><br>Live options data from Deribit</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-box"><b>⚡ Fast Analysis</b><br>Parallel processing & caching</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-box"><b>🎯 Professional</b><br>Black-76 model for crypto</div>', unsafe_allow_html=True)

# Run Analysis
if run_analysis and currencies and start_date < end_date:
    # Store configuration in session state
    st.session_state.currencies = currencies
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    st.session_state.risk_free_rate = risk_free_rate
    st.session_state.max_workers = max_workers

    # Progress container
    progress_container = st.container()

    with progress_container:
        st.markdown("### 🔄 Analysis in Progress...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Initialize data fetcher
            status_text.text("📡 Initializing data fetcher...")
            progress_bar.progress(10)

            fetcher = OptionsDataFetcher(
                currencies=currencies,
                start_date=start_date,
                end_date=end_date,
                cache_dir=str(Config.CACHE_DIR),
                max_workers=max_workers
            )
            st.session_state.data_fetcher = fetcher

            # Step 2: Fetch data
            status_text.text(f"📥 Fetching data for {', '.join(currencies)}...")
            progress_bar.progress(30)

            all_data = fetcher.fetch_all_data(use_cache=use_cache)
            st.session_state.raw_data = all_data

            # Step 3: Calculate Greeks
            status_text.text("🧮 Calculating Greeks...")
            progress_bar.progress(60)

            greeks_calculator = EnhancedGreeksCalculator(pricing_model=PricingModel.BLACK_76)

            processed_data = {}
            for currency, raw_df in all_data.items():
                if raw_df.empty:
                    st.warning(f"⚠️ No data available for {currency}")
                    continue

                # Calculate Greeks
                enhanced_df = greeks_calculator.calculate_greeks_dataframe(raw_df)
                processed_data[currency] = enhanced_df

                # Store in session state
                st.session_state[f"{currency}_data"] = enhanced_df

            # Step 4: Calculate Gamma Exposure
            status_text.text("⚡ Calculating Gamma Exposure...")
            progress_bar.progress(80)

            for currency, df in processed_data.items():
                if df.empty:
                    continue

                gex_analyzer = GammaExposureAnalyzer(df, currency)
                gex_df = gex_analyzer.calculate_gamma_exposure_by_strike()

                # Store GEX data
                st.session_state[f"{currency}_gex"] = gex_df
                st.session_state[f"{currency}_gex_analyzer"] = gex_analyzer

                # Detect gamma squeeze
                squeeze_indicators = gex_analyzer.detect_gamma_squeeze_risk(gex_df)
                st.session_state[f"{currency}_squeeze"] = squeeze_indicators

            # Step 5: Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)

            st.session_state.analysis_complete = True
            st.session_state.processed_data = processed_data

            # Success message
            st.success(f"""
            ✅ **Analysis Complete!**

            - Fetched data for: {', '.join(currencies)}
            - Date range: {start_date} to {end_date}
            - Total rows processed: {sum(len(df) for df in processed_data.values()):,}
            - Greeks calculated: ✓
            - Gamma Exposure analyzed: ✓
            """)

            # Display gamma squeeze warnings if any
            for currency in currencies:
                if f"{currency}_squeeze" in st.session_state:
                    squeeze = st.session_state[f"{currency}_squeeze"]
                    if squeeze.is_gamma_squeeze_risk:
                        st.warning(f"""
                        ⚠️ **{currency} Gamma Squeeze Risk Detected!**

                        - Squeeze Pressure Score: {squeeze.squeeze_pressure_score:.1f}/100
                        - Net Dealer Gamma: ${squeeze.net_dealer_gamma:,.0f}
                        - Gamma Flip Point: ${squeeze.gamma_flip_point:,.0f} ({squeeze.spot_to_flip_distance:+.1f}% from spot)
                        """)

            st.info("👈 **Navigate to analysis pages** using the sidebar menu to explore detailed visualizations!")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.exception("Analysis error")

            with st.expander("🐛 Error Details"):
                st.code(str(e))

# Display current session status
if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
    st.markdown("---")
    st.subheader("📋 Current Session Status")

    cols = st.columns(5)
    with cols[0]:
        st.metric("Currencies", ", ".join(st.session_state.currencies))
    with cols[1]:
        date_range = f"{st.session_state.start_date} to {st.session_state.end_date}"
        st.metric("Date Range", f"{(st.session_state.end_date - st.session_state.start_date).days}d")
    with cols[2]:
        st.metric("Risk-Free Rate", f"{st.session_state.risk_free_rate:.1%}")
    with cols[3]:
        total_rows = sum(len(st.session_state.get(f"{c}_data", pd.DataFrame())) for c in st.session_state.currencies)
        st.metric("Total Trades", f"{total_rows:,}")
    with cols[4]:
        st.metric("Status", "✅ Ready")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><b>BTC Options Analytics Dashboard</b> | Data: <a href='https://www.deribit.com' target='_blank'>Deribit</a> |
    Built with <a href='https://streamlit.io' target='_blank'>Streamlit</a> & <a href='https://plotly.com' target='_blank'>Plotly</a></p>
    <p>Source: <a href='https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit' target='_blank'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
