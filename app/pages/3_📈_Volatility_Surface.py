"""
Volatility Surface Analysis Page
3D IV surfaces, volatility skew, and term structure
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import griddata

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Volatility Surface", page_icon="📈", layout="wide")

st.title("📈 Implied Volatility Surface & Skew Analysis")
st.markdown("### Understanding market volatility expectations")

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
    help="Choose currency to analyze volatility"
)

# Get data
if f"{selected_currency}_data" not in st.session_state:
    st.error(f"No data available for {selected_currency}")
    st.stop()

data = st.session_state[f"{selected_currency}_data"]
spot_price = data['index_price'].iloc[-1]

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "🗻 3D Surface",
    "📊 IV Skew",
    "📉 Term Structure",
    "🔍 Analysis"
])

# === TAB 1: 3D VOLATILITY SURFACE ===
with tab1:
    st.subheader(f"{selected_currency} 3D Implied Volatility Surface")

    col1, col2 = st.columns(2)
    with col1:
        option_type_3d = st.selectbox(
            "Option Type for 3D Surface",
            ["call", "put", "both"],
            help="Select which option type to display"
        )
    with col2:
        surface_metric = st.selectbox(
            "Surface Metric",
            ["Implied Volatility", "Vega", "Gamma"],
            help="Choose which metric to visualize on surface"
        )

    # Filter data
    if option_type_3d == "both":
        surface_data = data.copy()
    else:
        surface_data = data[data['option_type'] == option_type_3d].copy()

    # Prepare data for surface
    surface_data = surface_data[
        (surface_data['time_to_maturity'] > 0) &
        (surface_data['iv'] > 0)
    ].copy()

    if not surface_data.empty:
        # Create grid for interpolation
        strike_range = np.linspace(
            surface_data['strike_price'].min(),
            surface_data['strike_price'].max(),
            50
        )
        ttm_range = np.linspace(
            surface_data['time_to_maturity'].min(),
            surface_data['time_to_maturity'].max(),
            50
        )

        # Select metric
        metric_map = {
            "Implied Volatility": 'iv',
            "Vega": 'vega',
            "Gamma": 'gamma'
        }
        metric_col = metric_map[surface_metric]

        # Create meshgrid
        X, Y = np.meshgrid(strike_range, ttm_range)

        # Interpolate
        Z = griddata(
            (surface_data['strike_price'], surface_data['time_to_maturity']),
            surface_data[metric_col],
            (X, Y),
            method='cubic',
            fill_value=np.nan
        )

        # Create 3D surface plot
        fig_3d = go.Figure(data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z if surface_metric == "Implied Volatility" else Z,
                colorscale='Viridis',
                hovertemplate=(
                    f"<b>{surface_metric}</b><br>" +
                    "Strike: $%{x:,.0f}<br>" +
                    "TTM: %{y:.1f} days<br>" +
                    f"{surface_metric}: " + "%{z:.4f}<br>" +
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title=surface_metric if surface_metric != "Implied Volatility" else "IV",
                    x=1.1
                )
            )
        ])

        # Add current spot price marker
        fig_3d.add_trace(
            go.Scatter3d(
                x=[spot_price] * len(ttm_range),
                y=ttm_range,
                z=[Z.max()] * len(ttm_range),
                mode='lines',
                line=dict(color='red', width=4),
                name=f'Spot: ${spot_price:,.0f}',
                hoverinfo='name'
            )
        )

        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Strike Price (USD)',
                yaxis_title='Time to Maturity (Days)',
                zaxis_title=surface_metric,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=700,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min IV", f"{surface_data['iv'].min()*100:.2f}%")
        with col2:
            st.metric("Max IV", f"{surface_data['iv'].max()*100:.2f}%")
        with col3:
            st.metric("Mean IV", f"{surface_data['iv'].mean()*100:.2f}%")
        with col4:
            st.metric("Std Dev", f"{surface_data['iv'].std()*100:.2f}%")

    else:
        st.warning("Insufficient data for 3D surface plot")

# === TAB 2: IV SKEW ===
with tab2:
    st.subheader("📊 Implied Volatility Skew by Maturity")

    # Select maturity buckets
    maturity_buckets = st.multiselect(
        "Select Maturity Ranges (days)",
        ["0-7", "7-14", "14-30", "30-60", "60-90", "90-180", "180+"],
        default=["7-14", "30-60", "90-180"],
        help="Choose maturity ranges to display"
    )

    # Create maturity bins
    def assign_bucket(ttm):
        if ttm <= 7:
            return "0-7"
        elif ttm <= 14:
            return "7-14"
        elif ttm <= 30:
            return "14-30"
        elif ttm <= 60:
            return "30-60"
        elif ttm <= 90:
            return "60-90"
        elif ttm <= 180:
            return "90-180"
        else:
            return "180+"

    data['ttm_bucket'] = data['time_to_maturity'].apply(assign_bucket)

    # Filter by selected buckets
    filtered_data = data[data['ttm_bucket'].isin(maturity_buckets)].copy()

    if not filtered_data.empty:
        # Create skew plots
        fig_skew = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Call IV Skew", "Put IV Skew")
        )

        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']

        for idx, bucket in enumerate(maturity_buckets):
            bucket_data_calls = filtered_data[
                (filtered_data['ttm_bucket'] == bucket) &
                (filtered_data['option_type'] == 'call')
            ].sort_values('strike_price')

            bucket_data_puts = filtered_data[
                (filtered_data['ttm_bucket'] == bucket) &
                (filtered_data['option_type'] == 'put')
            ].sort_values('strike_price')

            color = colors[idx % len(colors)]

            # Calls
            if not bucket_data_calls.empty:
                fig_skew.add_trace(
                    go.Scatter(
                        x=bucket_data_calls['strike_price'],
                        y=bucket_data_calls['iv'] * 100,
                        mode='lines+markers',
                        name=f'{bucket}d',
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        hovertemplate=(
                            f"<b>{bucket} days</b><br>" +
                            "Strike: $%{x:,.0f}<br>" +
                            "IV: %{y:.2f}%<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=1, col=1
                )

            # Puts
            if not bucket_data_puts.empty:
                fig_skew.add_trace(
                    go.Scatter(
                        x=bucket_data_puts['strike_price'],
                        y=bucket_data_puts['iv'] * 100,
                        mode='lines+markers',
                        name=f'{bucket}d',
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{bucket} days</b><br>" +
                            "Strike: $%{x:,.0f}<br>" +
                            "IV: %{y:.2f}%<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=1, col=2
                )

        # Add spot price lines
        fig_skew.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="yellow",
            line_width=2,
            annotation_text=f"Spot: ${spot_price:,.0f}",
            annotation_position="top",
            row=1, col=1
        )
        fig_skew.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="yellow",
            line_width=2,
            annotation_text=f"Spot: ${spot_price:,.0f}",
            annotation_position="top",
            row=1, col=2
        )

        fig_skew.update_layout(
            height=600,
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        fig_skew.update_xaxes(title_text="Strike Price (USD)", row=1, col=1)
        fig_skew.update_xaxes(title_text="Strike Price (USD)", row=1, col=2)
        fig_skew.update_yaxes(title_text="Implied Volatility (%)", row=1, col=1)
        fig_skew.update_yaxes(title_text="Implied Volatility (%)", row=1, col=2)

        st.plotly_chart(fig_skew, use_container_width=True)

        # Skew interpretation
        with st.expander("📖 Understanding IV Skew"):
            st.markdown("""
            **Volatility Skew Patterns:**

            - **Positive Skew (Smile)**: Higher IV for OTM options
              - Common in crypto markets
              - Indicates demand for tail risk protection

            - **Negative Skew (Smirk)**: Higher IV for OTM puts
              - Common in equity markets
              - Fear of downside moves

            - **Flat Skew**: Similar IV across strikes
              - Balanced market expectations
              - Less directional bias

            **What to Look For:**
            - Steepening skew: Increased fear/uncertainty
            - Flattening skew: Market calming down
            - ATM vs OTM spreads: Premium for tail risk
            """)
    else:
        st.warning("No data available for selected maturity ranges")

# === TAB 3: TERM STRUCTURE ===
with tab3:
    st.subheader("📉 Volatility Term Structure")

    # ATM options only
    atm_threshold = 0.05  # Within 5% of ATM
    atm_data = data[
        (data['moneyness'] >= 1 - atm_threshold) &
        (data['moneyness'] <= 1 + atm_threshold)
    ].copy()

    if not atm_data.empty:
        # Group by maturity date
        term_structure = atm_data.groupby(['maturity_date', 'option_type']).agg({
            'iv': 'mean',
            'time_to_maturity': 'mean',
            'volume_btc': 'sum'
        }).reset_index()

        term_structure = term_structure.sort_values('time_to_maturity')

        # Create term structure chart
        fig_term = go.Figure()

        for option_type, color in [('call', '#00CC96'), ('put', '#EF553B')]:
            type_data = term_structure[term_structure['option_type'] == option_type]
            fig_term.add_trace(
                go.Scatter(
                    x=type_data['time_to_maturity'],
                    y=type_data['iv'] * 100,
                    mode='lines+markers',
                    name=option_type.capitalize(),
                    line=dict(color=color, width=3),
                    marker=dict(
                        size=type_data['volume_btc'] / type_data['volume_btc'].max() * 30 + 5,
                        color=color
                    ),
                    hovertemplate=(
                        f"<b>{option_type.capitalize()}</b><br>" +
                        "TTM: %{x:.1f} days<br>" +
                        "IV: %{y:.2f}%<br>" +
                        "<extra></extra>"
                    )
                )
            )

        fig_term.update_layout(
            title="ATM Implied Volatility Term Structure",
            xaxis_title="Time to Maturity (Days)",
            yaxis_title="Implied Volatility (%)",
            height=500,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_term, use_container_width=True)

        # Term structure interpretation
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Call Term Structure")
            call_term = term_structure[term_structure['option_type'] == 'call']
            if len(call_term) >= 2:
                short_term_iv = call_term.iloc[0]['iv'] * 100
                long_term_iv = call_term.iloc[-1]['iv'] * 100
                term_slope = long_term_iv - short_term_iv

                if term_slope > 0:
                    st.success(f"📈 Upward sloping (+{term_slope:.2f}%)")
                    st.caption("Market expects higher volatility in future")
                else:
                    st.warning(f"📉 Downward sloping ({term_slope:.2f}%)")
                    st.caption("Near-term volatility elevated (possible event)")

        with col2:
            st.markdown("#### Put Term Structure")
            put_term = term_structure[term_structure['option_type'] == 'put']
            if len(put_term) >= 2:
                short_term_iv = put_term.iloc[0]['iv'] * 100
                long_term_iv = put_term.iloc[-1]['iv'] * 100
                term_slope = long_term_iv - short_term_iv

                if term_slope > 0:
                    st.success(f"📈 Upward sloping (+{term_slope:.2f}%)")
                    st.caption("Market expects higher volatility in future")
                else:
                    st.warning(f"📉 Downward sloping ({term_slope:.2f}%)")
                    st.caption("Near-term volatility elevated (possible event)")

        with st.expander("📖 Understanding Term Structure"):
            st.markdown("""
            **Volatility Term Structure Patterns:**

            - **Upward Sloping (Contango)**:
              - Normal market conditions
              - Future uncertainty priced higher
              - Long-term vol > short-term vol

            - **Downward Sloping (Backwardation)**:
              - Event-driven markets
              - Near-term risk elevated
              - Short-term vol > long-term vol

            - **Humped**:
              - Specific event expected at intermediate term
              - Vol peaks in middle maturities

            **Trading Implications:**
            - Backwardation: Potential calendar spread opportunities
            - Contango: Neutral to bullish on volatility decay
            - Flat: Market equilibrium, limited term structure trades
            """)
    else:
        st.warning("Insufficient ATM data for term structure analysis")

# === TAB 4: ANALYSIS ===
with tab4:
    st.subheader("🔍 Volatility Analysis & Insights")

    # Calculate key metrics
    call_iv_mean = data[data['option_type'] == 'call']['iv'].mean()
    put_iv_mean = data[data['option_type'] == 'put']['iv'].mean()
    iv_spread = (call_iv_mean - put_iv_mean) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Call IV",
            f"{call_iv_mean*100:.2f}%",
            help="Volume-weighted average IV for calls"
        )

    with col2:
        st.metric(
            "Average Put IV",
            f"{put_iv_mean*100:.2f}%",
            help="Volume-weighted average IV for puts"
        )

    with col3:
        st.metric(
            "Call-Put IV Spread",
            f"{iv_spread:+.2f}%",
            delta="Calls higher" if iv_spread > 0 else "Puts higher",
            help="Difference in average IV between calls and puts"
        )

    # IV distribution histogram
    st.markdown("---")
    st.markdown("#### Implied Volatility Distribution")

    fig_dist = go.Figure()

    for option_type, color in [('call', '#00CC96'), ('put', '#EF553B')]:
        type_data = data[data['option_type'] == option_type]
        fig_dist.add_trace(
            go.Histogram(
                x=type_data['iv'] * 100,
                name=option_type.capitalize(),
                marker_color=color,
                opacity=0.7,
                nbinsx=50,
                hovertemplate=(
                    f"<b>{option_type.capitalize()}</b><br>" +
                    "IV Range: %{x:.2f}%<br>" +
                    "Count: %{y}<br>" +
                    "<extra></extra>"
                )
            )
        )

    fig_dist.update_layout(
        title="IV Distribution Across All Options",
        xaxis_title="Implied Volatility (%)",
        yaxis_title="Frequency",
        height=400,
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # Volatility percentiles
    st.markdown("---")
    st.markdown("#### Volatility Percentiles")

    percentiles = [10, 25, 50, 75, 90]
    call_percentiles = np.percentile(data[data['option_type'] == 'call']['iv'] * 100, percentiles)
    put_percentiles = np.percentile(data[data['option_type'] == 'put']['iv'] * 100, percentiles)

    percentile_df = pd.DataFrame({
        'Percentile': [f'{p}%' for p in percentiles],
        'Call IV': [f'{v:.2f}%' for v in call_percentiles],
        'Put IV': [f'{v:.2f}%' for v in put_percentiles]
    })

    st.dataframe(percentile_df, use_container_width=True)

st.markdown("---")
st.info("""
**💡 Pro Tip**: Watch for changes in volatility skew and term structure.
Sudden steepening or flattening can signal market regime changes or upcoming events.
""")
