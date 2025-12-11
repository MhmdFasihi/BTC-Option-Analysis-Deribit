"""
Gamma Exposure (GEX) Analysis Page
CRITICAL: Gamma squeeze detection and market maker positioning
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analytics.gamma_exposure import GammaExposureAnalyzer

st.set_page_config(page_title="Gamma Exposure", page_icon="⚡", layout="wide")

st.title("⚡ Gamma Exposure & Squeeze Analysis")
st.markdown("### Understanding market maker positioning and gamma squeeze dynamics")

# Check if data is loaded
if 'analysis_complete' not in st.session_state or not st.session_state.analysis_complete:
    st.warning("⚠️ No analysis data available. Please run analysis from the main page first.")
    st.info("👈 Go to the main page and click '🚀 Run Analysis' to get started")
    st.stop()

# Get currencies
currencies = st.session_state.currencies

# Currency selector
selected_currency = st.selectbox(
    "Select Currency",
    currencies,
    help="Choose currency to analyze gamma exposure"
)

# Get data
if f"{selected_currency}_data" not in st.session_state:
    st.error(f"No data available for {selected_currency}")
    st.stop()

data = st.session_state[f"{selected_currency}_data"]
gex_df = st.session_state.get(f"{selected_currency}_gex", pd.DataFrame())
gex_analyzer = st.session_state.get(f"{selected_currency}_gex_analyzer")
squeeze_indicators = st.session_state.get(f"{selected_currency}_squeeze")

spot_price = data['index_price'].iloc[-1]

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 GEX Overview",
    "🎯 Strike Analysis",
    "⚠️ Gamma Squeeze",
    "📈 Market Dynamics"
])

# === TAB 1: GEX OVERVIEW ===
with tab1:
    st.subheader(f"{selected_currency} Gamma Exposure Overview")

    if gex_df.empty:
        st.warning("No gamma exposure data available")
    else:
        # Summary metrics
        calls_gex = gex_df[gex_df['option_type'] == 'call']
        puts_gex = gex_df[gex_df['option_type'] == 'put']
        net_gex_data = gex_df[gex_df['option_type'] == 'net']

        total_call_gex = calls_gex['gex'].sum() if not calls_gex.empty else 0
        total_put_gex = puts_gex['gex'].sum() if not puts_gex.empty else 0
        total_net_gex = net_gex_data['gex'].sum() if not net_gex_data.empty else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Spot Price",
                f"${spot_price:,.0f}",
                help="Current underlying asset price"
            )

        with col2:
            st.metric(
                "Total Call GEX",
                f"${total_call_gex:,.0f}",
                delta="Positive Gamma",
                delta_color="normal",
                help="Positive = dealers buy on rallies, sell on dips (stabilizing)"
            )

        with col3:
            st.metric(
                "Total Put GEX",
                f"${total_put_gex:,.0f}",
                delta="Negative Gamma",
                delta_color="inverse",
                help="Negative = dealers sell on rallies, buy on dips (destabilizing)"
            )

        with col4:
            st.metric(
                "Net GEX",
                f"${total_net_gex:,.0f}",
                delta="Overall Positioning",
                help="Net dealer gamma exposure"
            )

        # Main GEX Chart
        st.markdown("---")
        st.subheader("📊 Gamma Exposure by Strike Price")

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{selected_currency} Gamma Exposure Profile",
                "Net GEX (Calls + Puts)"
            )
        )

        # Top chart: Calls and Puts separately
        if not calls_gex.empty:
            fig.add_trace(
                go.Bar(
                    x=calls_gex['strike_price'],
                    y=calls_gex['gex'],
                    name='Call GEX',
                    marker_color='#00CC96',
                    opacity=0.7,
                    hovertemplate=(
                        "<b>Call GEX</b><br>" +
                        "Strike: $%{x:,.0f}<br>" +
                        "GEX: $%{y:,.0f}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=1
            )

        if not puts_gex.empty:
            fig.add_trace(
                go.Bar(
                    x=puts_gex['strike_price'],
                    y=puts_gex['gex'],
                    name='Put GEX',
                    marker_color='#EF553B',
                    opacity=0.7,
                    hovertemplate=(
                        "<b>Put GEX</b><br>" +
                        "Strike: $%{x:,.0f}<br>" +
                        "GEX: $%{y:,.0f}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=1
            )

        # Bottom chart: Net GEX
        if not net_gex_data.empty:
            # Color based on positive/negative
            colors = ['#00CC96' if x > 0 else '#EF553B' for x in net_gex_data['gex']]

            fig.add_trace(
                go.Bar(
                    x=net_gex_data['strike_price'],
                    y=net_gex_data['gex'],
                    name='Net GEX',
                    marker_color=colors,
                    opacity=0.8,
                    hovertemplate=(
                        "<b>Net GEX</b><br>" +
                        "Strike: $%{x:,.0f}<br>" +
                        "Net GEX: $%{y:,.0f}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=2, col=1
            )

        # Add current price line
        fig.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="yellow",
            line_width=2,
            annotation_text=f"Spot: ${spot_price:,.0f}",
            annotation_position="top"
        )

        # Add gamma flip point if available
        if squeeze_indicators and squeeze_indicators.gamma_flip_point:
            fig.add_vline(
                x=squeeze_indicators.gamma_flip_point,
                line_dash="dot",
                line_color="cyan",
                line_width=2,
                annotation_text=f"Flip: ${squeeze_indicators.gamma_flip_point:,.0f}",
                annotation_position="bottom"
            )

        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        fig.update_xaxes(title_text="Strike Price (USD)", row=2, col=1)
        fig.update_yaxes(title_text="Gamma Exposure ($)", row=1, col=1)
        fig.update_yaxes(title_text="Net GEX ($)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Explanation
        with st.expander("📖 Understanding Gamma Exposure"):
            st.markdown("""
            **What is Gamma Exposure (GEX)?**

            GEX represents the hedging flow that market makers must execute when the underlying price moves.

            **Interpretation:**
            - **Positive GEX (Green)**: Market makers buy into rallies and sell into dips → **Stabilizing** (price support/resistance)
            - **Negative GEX (Red)**: Market makers sell into rallies and buy into dips → **Destabilizing** (breakout potential)

            **Key Levels:**
            - **Yellow Line**: Current spot price
            - **Cyan Line**: Gamma flip point (where GEX changes sign)
            - **High Positive GEX**: Strong support/resistance
            - **High Negative GEX**: Potential breakout zone
            """)

# === TAB 2: STRIKE ANALYSIS ===
with tab2:
    st.subheader("🎯 Strike-by-Strike Analysis")

    if not net_gex_data.empty:
        # Filter range
        col1, col2 = st.columns(2)
        with col1:
            min_strike = st.number_input(
                "Min Strike",
                value=float(spot_price * 0.8),
                step=1000.0,
                help="Minimum strike to display"
            )
        with col2:
            max_strike = st.number_input(
                "Max Strike",
                value=float(spot_price * 1.2),
                step=1000.0,
                help="Maximum strike to display"
            )

        # Filter data
        filtered_gex = net_gex_data[
            (net_gex_data['strike_price'] >= min_strike) &
            (net_gex_data['strike_price'] <= max_strike)
        ].copy()

        if not filtered_gex.empty:
            # Sort by absolute GEX
            filtered_gex['abs_gex'] = filtered_gex['gex'].abs()
            top_strikes = filtered_gex.nlargest(20, 'abs_gex')

            # Display table
            st.dataframe(
                top_strikes[[
                    'strike_price', 'gex', 'distance_from_spot_pct'
                ]].rename(columns={
                    'strike_price': 'Strike Price',
                    'gex': 'Net GEX ($)',
                    'distance_from_spot_pct': 'Distance from Spot (%)'
                }).style.format({
                    'Strike Price': '${:,.0f}',
                    'Net GEX ($)': '${:,.0f}',
                    'Distance from Spot (%)': '{:+.2f}%'
                }).background_gradient(subset=['Net GEX ($)'], cmap='RdYlGn'),
                use_container_width=True
            )

            # Download button
            csv = top_strikes.to_csv(index=False)
            st.download_button(
                "📥 Download GEX Data (CSV)",
                csv,
                f"{selected_currency}_gex_analysis.csv",
                "text/csv",
                key='download-gex'
            )

# === TAB 3: GAMMA SQUEEZE ===
with tab3:
    st.subheader("⚠️ Gamma Squeeze Risk Analysis")

    if squeeze_indicators:
        # Squeeze risk indicator
        if squeeze_indicators.is_gamma_squeeze_risk:
            st.error(f"""
            🚨 **GAMMA SQUEEZE RISK DETECTED FOR {selected_currency}**

            Market conditions indicate potential for gamma squeeze dynamics.
            """)
        else:
            st.success(f"✅ No significant gamma squeeze risk detected for {selected_currency}")

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            pressure_color = "🔴" if squeeze_indicators.squeeze_pressure_score > 70 else "🟡" if squeeze_indicators.squeeze_pressure_score > 40 else "🟢"
            st.metric(
                "Squeeze Pressure Score",
                f"{pressure_color} {squeeze_indicators.squeeze_pressure_score:.1f}/100",
                help="Higher = more squeeze risk"
            )

        with col2:
            st.metric(
                "Net Dealer Gamma",
                f"${squeeze_indicators.net_dealer_gamma:,.0f}",
                delta="Short Gamma" if squeeze_indicators.net_dealer_gamma < 0 else "Long Gamma",
                delta_color="inverse" if squeeze_indicators.net_dealer_gamma < 0 else "normal",
                help="Negative = dealers short gamma (squeeze potential)"
            )

        with col3:
            if squeeze_indicators.gamma_flip_point:
                st.metric(
                    "Gamma Flip Point",
                    f"${squeeze_indicators.gamma_flip_point:,.0f}",
                    delta=f"{squeeze_indicators.spot_to_flip_distance:+.1f}% from spot",
                    help="Price where net GEX = 0"
                )

        # Gauge chart for pressure score
        st.markdown("---")
        st.subheader("Squeeze Pressure Gauge")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=squeeze_indicators.squeeze_pressure_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Squeeze Pressure Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#00CC96'},
                    {'range': [40, 70], 'color': '#FFA500'},
                    {'range': [70, 100], 'color': '#EF553B'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig_gauge.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Arial"}
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        # Explanation
        with st.expander("📖 What is a Gamma Squeeze?"):
            st.markdown("""
            **Gamma Squeeze Mechanics:**

            1. **Setup**: Large call option buying → dealers sell calls → dealers are SHORT gamma
            2. **Trigger**: Price moves up → dealers' negative gamma forces them to buy stock to hedge
            3. **Acceleration**: Dealer buying pushes price higher → triggers more hedging → feedback loop
            4. **Result**: Explosive upward price movement (or downward if put-driven)

            **Risk Factors:**
            - **Dealer Short Gamma**: Negative net dealer position
            - **Proximity to Flip Point**: Close to gamma neutral level
            - **Concentration**: Large negative GEX near current price
            - **Volume**: High option trading volume in OTM strikes

            **Score Interpretation:**
            - **0-40**: Low risk (green)
            - **40-70**: Moderate risk (yellow)
            - **70-100**: High risk (red)
            """)

# === TAB 4: MARKET DYNAMICS ===
with tab4:
    st.subheader("📈 Market Maker Positioning & Dynamics")

    if not gex_df.empty:
        # Scenario analysis
        st.markdown("### Price Movement Scenarios")

        price_changes = [-10, -5, -2, 0, 2, 5, 10]
        scenarios = []

        for pct_change in price_changes:
            new_price = spot_price * (1 + pct_change / 100)
            # Find GEX at this level (interpolate if needed)
            nearest_strike = net_gex_data.iloc[(net_gex_data['strike_price'] - new_price).abs().argsort()[:1]]

            if not nearest_strike.empty:
                gex_at_price = nearest_strike['gex'].values[0]
                scenarios.append({
                    'Price Change (%)': pct_change,
                    'New Price': new_price,
                    'GEX at Level': gex_at_price,
                    'MM Flow': 'Buy' if gex_at_price > 0 else 'Sell',
                    'Effect': 'Stabilizing' if gex_at_price > 0 else 'Destabilizing'
                })

        scenario_df = pd.DataFrame(scenarios)

        if not scenario_df.empty:
            st.dataframe(
                scenario_df.style.format({
                    'Price Change (%)': '{:+.0f}%',
                    'New Price': '${:,.0f}',
                    'GEX at Level': '${:,.0f}'
                }).applymap(
                    lambda x: 'background-color: #d4edda' if x == 'Stabilizing' else ('background-color: #f8d7da' if x == 'Destabilizing' else ''),
                    subset=['Effect']
                ),
                use_container_width=True
            )

        # Pin risk analysis
        st.markdown("---")
        st.subheader("📌 Pin Risk Analysis")

        if gex_analyzer:
            pin_analysis = gex_analyzer.analyze_pin_risk(gex_df)

            if pin_analysis['has_pin_risk']:
                st.warning(f"⚠️ Pin risk detected! Price may be drawn to strikes: {', '.join([f'${s:,.0f}' for s in pin_analysis['near_price_pins']])}")
            else:
                st.success("✅ No significant pin risk detected")

            st.markdown(f"""
            **Pin Strikes (High Positive GEX):**
            {', '.join([f'${s:,.0f}' for s in pin_analysis['pin_strikes']])}

            Strikes with very high positive GEX can act as "magnets" for price near expiration.
            """)

st.markdown("---")
st.info("""
**💡 Pro Tip**: Watch for changes in GEX levels as price moves.
Crossing from positive to negative GEX (or vice versa) can signal regime changes in market behavior.
""")
