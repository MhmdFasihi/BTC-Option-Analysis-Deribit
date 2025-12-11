"""
Market Overview Page
Comprehensive market summary, volume analysis, and active strikes
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

st.set_page_config(page_title="Market Overview", page_icon="📊", layout="wide")

st.title("📊 Market Overview & Volume Analysis")
st.markdown("### Comprehensive view of options market activity")

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
    help="Choose currency to analyze"
)

# Get data
if f"{selected_currency}_data" not in st.session_state:
    st.error(f"No data available for {selected_currency}")
    st.stop()

data = st.session_state[f"{selected_currency}_data"]

# === MARKET SUMMARY METRICS ===
st.subheader(f"{selected_currency} Market Summary")

# Calculate key metrics
spot_price = data['index_price'].iloc[-1]
total_volume_btc = data['volume_btc'].sum()
total_volume_usd = data['volume_usd'].sum()
total_contracts = data['contracts'].sum()

# Call vs Put metrics
calls_data = data[data['option_type'] == 'call']
puts_data = data[data['option_type'] == 'put']

call_volume_btc = calls_data['volume_btc'].sum()
put_volume_btc = puts_data['volume_btc'].sum()
call_put_ratio = call_volume_btc / put_volume_btc if put_volume_btc > 0 else 0

call_contracts = calls_data['contracts'].sum()
put_contracts = puts_data['contracts'].sum()

avg_iv_calls = calls_data['iv'].mean() if not calls_data.empty else 0
avg_iv_puts = puts_data['iv'].mean() if not puts_data.empty else 0

# Display metrics in columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Price",
        f"${spot_price:,.0f}",
        help="Current underlying asset price"
    )

with col2:
    st.metric(
        "Total Volume",
        f"{total_volume_btc:,.2f} {selected_currency}",
        delta=f"${total_volume_usd:,.0f}",
        help="Total options trading volume"
    )

with col3:
    st.metric(
        "Call/Put Ratio",
        f"{call_put_ratio:.2f}",
        delta="Bullish" if call_put_ratio > 1 else "Bearish",
        delta_color="normal" if call_put_ratio > 1 else "inverse",
        help="Volume ratio of calls to puts"
    )

with col4:
    st.metric(
        "Total Contracts",
        f"{total_contracts:,.0f}",
        help="Total number of contracts traded"
    )

with col5:
    st.metric(
        "Avg IV (C/P)",
        f"{avg_iv_calls*100:.1f}% / {avg_iv_puts*100:.1f}%",
        help="Average implied volatility for calls and puts"
    )

# === VOLUME ANALYSIS ===
st.markdown("---")
st.subheader("📈 Volume Analysis")

# Daily volume aggregation
data['date'] = pd.to_datetime(data['date_time']).dt.date
daily_volume = data.groupby(['date', 'option_type']).agg({
    'volume_btc': 'sum',
    'volume_usd': 'sum',
    'contracts': 'sum'
}).reset_index()

# Create volume chart
fig_volume = make_subplots(
    rows=2, cols=1,
    row_heights=[0.6, 0.4],
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        f"{selected_currency} Daily Volume (BTC)",
        "Call/Put Volume Comparison"
    )
)

# Top chart: Stacked volume by option type
for option_type, color in [('call', '#00CC96'), ('put', '#EF553B')]:
    type_data = daily_volume[daily_volume['option_type'] == option_type]
    fig_volume.add_trace(
        go.Bar(
            x=type_data['date'],
            y=type_data['volume_btc'],
            name=option_type.capitalize(),
            marker_color=color,
            hovertemplate=(
                f"<b>{option_type.capitalize()}</b><br>" +
                "Date: %{x}<br>" +
                "Volume: %{y:,.2f} BTC<br>" +
                "<extra></extra>"
            )
        ),
        row=1, col=1
    )

# Bottom chart: Call/Put ratio over time
call_daily = daily_volume[daily_volume['option_type'] == 'call'].set_index('date')['volume_btc']
put_daily = daily_volume[daily_volume['option_type'] == 'put'].set_index('date')['volume_btc']
cp_ratio_daily = (call_daily / put_daily).fillna(0)

fig_volume.add_trace(
    go.Scatter(
        x=cp_ratio_daily.index,
        y=cp_ratio_daily.values,
        mode='lines+markers',
        name='C/P Ratio',
        line=dict(color='#FFA500', width=2),
        marker=dict(size=6),
        hovertemplate=(
            "<b>Call/Put Ratio</b><br>" +
            "Date: %{x}<br>" +
            "Ratio: %{y:.2f}<br>" +
            "<extra></extra>"
        )
    ),
    row=2, col=1
)

# Add horizontal line at 1.0 (neutral)
fig_volume.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="gray",
    row=2, col=1,
    annotation_text="Neutral (1.0)",
    annotation_position="right"
)

fig_volume.update_layout(
    height=700,
    showlegend=True,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    barmode='stack'
)

fig_volume.update_xaxes(title_text="Date", row=2, col=1)
fig_volume.update_yaxes(title_text="Volume (BTC)", row=1, col=1)
fig_volume.update_yaxes(title_text="C/P Ratio", row=2, col=1)

st.plotly_chart(fig_volume, use_container_width=True)

# === MOST ACTIVE STRIKES ===
st.markdown("---")
st.subheader("🎯 Most Active Strikes")

col1, col2 = st.columns(2)

# Aggregate by strike
strike_volume = data.groupby(['strike_price', 'option_type']).agg({
    'volume_btc': 'sum',
    'volume_usd': 'sum',
    'contracts': 'sum',
    'iv': 'mean'
}).reset_index()

# Add distance from spot
strike_volume['distance_from_spot_pct'] = ((strike_volume['strike_price'] - spot_price) / spot_price) * 100

with col1:
    st.markdown("#### Top Call Strikes by Volume")
    top_calls = strike_volume[strike_volume['option_type'] == 'call'].nlargest(10, 'volume_btc')

    if not top_calls.empty:
        st.dataframe(
            top_calls[['strike_price', 'volume_btc', 'contracts', 'iv', 'distance_from_spot_pct']].rename(columns={
                'strike_price': 'Strike',
                'volume_btc': f'Volume ({selected_currency})',
                'contracts': 'Contracts',
                'iv': 'Avg IV',
                'distance_from_spot_pct': 'From Spot (%)'
            }).style.format({
                'Strike': '${:,.0f}',
                f'Volume ({selected_currency})': '{:.2f}',
                'Contracts': '{:.0f}',
                'Avg IV': '{:.2%}',
                'From Spot (%)': '{:+.2f}%'
            }).background_gradient(subset=[f'Volume ({selected_currency})'], cmap='Greens'),
            use_container_width=True,
            height=400
        )
    else:
        st.info("No call data available")

with col2:
    st.markdown("#### Top Put Strikes by Volume")
    top_puts = strike_volume[strike_volume['option_type'] == 'put'].nlargest(10, 'volume_btc')

    if not top_puts.empty:
        st.dataframe(
            top_puts[['strike_price', 'volume_btc', 'contracts', 'iv', 'distance_from_spot_pct']].rename(columns={
                'strike_price': 'Strike',
                'volume_btc': f'Volume ({selected_currency})',
                'contracts': 'Contracts',
                'iv': 'Avg IV',
                'distance_from_spot_pct': 'From Spot (%)'
            }).style.format({
                'Strike': '${:,.0f}',
                f'Volume ({selected_currency})': '{:.2f}',
                'Contracts': '{:.0f}',
                'Avg IV': '{:.2%}',
                'From Spot (%)': '{:+.2f}%'
            }).background_gradient(subset=[f'Volume ({selected_currency})'], cmap='Reds'),
            use_container_width=True,
            height=400
        )
    else:
        st.info("No put data available")

# === MATURITY DISTRIBUTION ===
st.markdown("---")
st.subheader("📅 Maturity Distribution")

# Group by maturity date
maturity_analysis = data.groupby(['maturity_date', 'option_type']).agg({
    'volume_btc': 'sum',
    'contracts': 'sum',
    'strike_price': 'count'
}).reset_index()

maturity_analysis.rename(columns={'strike_price': 'num_strikes'}, inplace=True)
maturity_analysis['maturity_date'] = pd.to_datetime(maturity_analysis['maturity_date'])

# Create maturity chart
fig_maturity = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Volume by Maturity Date",
        "Number of Active Strikes"
    )
)

# Chart 1: Volume by maturity
for option_type, color in [('call', '#00CC96'), ('put', '#EF553B')]:
    type_data = maturity_analysis[maturity_analysis['option_type'] == option_type]
    fig_maturity.add_trace(
        go.Bar(
            x=type_data['maturity_date'],
            y=type_data['volume_btc'],
            name=option_type.capitalize(),
            marker_color=color,
            hovertemplate=(
                f"<b>{option_type.capitalize()}</b><br>" +
                "Maturity: %{x|%Y-%m-%d}<br>" +
                "Volume: %{y:,.2f} BTC<br>" +
                "<extra></extra>"
            )
        ),
        row=1, col=1
    )

# Chart 2: Number of strikes
for option_type, color in [('call', '#00CC96'), ('put', '#EF553B')]:
    type_data = maturity_analysis[maturity_analysis['option_type'] == option_type]
    fig_maturity.add_trace(
        go.Bar(
            x=type_data['maturity_date'],
            y=type_data['num_strikes'],
            name=option_type.capitalize(),
            marker_color=color,
            showlegend=False,
            hovertemplate=(
                f"<b>{option_type.capitalize()}</b><br>" +
                "Maturity: %{x|%Y-%m-%d}<br>" +
                "Strikes: %{y}<br>" +
                "<extra></extra>"
            )
        ),
        row=1, col=2
    )

fig_maturity.update_layout(
    height=500,
    showlegend=True,
    barmode='group',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig_maturity.update_xaxes(title_text="Maturity Date", row=1, col=1)
fig_maturity.update_xaxes(title_text="Maturity Date", row=1, col=2)
fig_maturity.update_yaxes(title_text="Volume (BTC)", row=1, col=1)
fig_maturity.update_yaxes(title_text="Number of Strikes", row=1, col=2)

st.plotly_chart(fig_maturity, use_container_width=True)

# === TIME TO MATURITY DISTRIBUTION ===
st.markdown("---")
st.subheader("⏰ Time to Maturity Distribution")

# Bin time to maturity
data['ttm_bins'] = pd.cut(
    data['time_to_maturity'],
    bins=[0, 7, 14, 30, 60, 90, 180, 365],
    labels=['0-7d', '7-14d', '14-30d', '30-60d', '60-90d', '90-180d', '180-365d']
)

ttm_volume = data.groupby(['ttm_bins', 'option_type'])['volume_btc'].sum().reset_index()

fig_ttm = go.Figure()

for option_type, color in [('call', '#00CC96'), ('put', '#EF553B')]:
    type_data = ttm_volume[ttm_volume['option_type'] == option_type]
    fig_ttm.add_trace(
        go.Bar(
            x=type_data['ttm_bins'].astype(str),
            y=type_data['volume_btc'],
            name=option_type.capitalize(),
            marker_color=color,
            hovertemplate=(
                f"<b>{option_type.capitalize()}</b><br>" +
                "TTM Range: %{x}<br>" +
                "Volume: %{y:,.2f} BTC<br>" +
                "<extra></extra>"
            )
        )
    )

fig_ttm.update_layout(
    title="Volume Distribution by Time to Maturity",
    xaxis_title="Time to Maturity",
    yaxis_title="Volume (BTC)",
    height=400,
    barmode='group',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_ttm, use_container_width=True)

# === MONEYNESS ANALYSIS ===
st.markdown("---")
st.subheader("💰 Moneyness Distribution")

# Bin moneyness
data['moneyness_bins'] = pd.cut(
    data['moneyness'],
    bins=[0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 2.0],
    labels=['Deep OTM', 'OTM', 'Slightly OTM', 'ATM', 'Slightly ITM', 'ITM', 'Deep ITM', 'Very Deep ITM']
)

moneyness_volume = data.groupby(['moneyness_bins', 'option_type'])['volume_btc'].sum().reset_index()

fig_moneyness = go.Figure()

for option_type, color in [('call', '#00CC96'), ('put', '#EF553B')]:
    type_data = moneyness_volume[moneyness_volume['option_type'] == option_type]
    fig_moneyness.add_trace(
        go.Bar(
            x=type_data['moneyness_bins'].astype(str),
            y=type_data['volume_btc'],
            name=option_type.capitalize(),
            marker_color=color,
            hovertemplate=(
                f"<b>{option_type.capitalize()}</b><br>" +
                "Moneyness: %{x}<br>" +
                "Volume: %{y:,.2f} BTC<br>" +
                "<extra></extra>"
            )
        )
    )

fig_moneyness.update_layout(
    title="Volume Distribution by Moneyness",
    xaxis_title="Moneyness (Spot/Strike)",
    yaxis_title="Volume (BTC)",
    height=400,
    barmode='group',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_moneyness, use_container_width=True)

# Explanation
with st.expander("📖 Understanding Market Overview Metrics"):
    st.markdown("""
    **Key Metrics:**

    - **Call/Put Ratio**: Ratio of call volume to put volume
      - > 1.0: More call activity (bullish sentiment)
      - < 1.0: More put activity (bearish sentiment)
      - = 1.0: Neutral sentiment

    - **Moneyness**: Ratio of spot price to strike price
      - ATM (At-The-Money): Spot ≈ Strike (0.95 - 1.05)
      - ITM (In-The-Money): Calls when Spot > Strike, Puts when Spot < Strike
      - OTM (Out-of-The-Money): Calls when Spot < Strike, Puts when Spot > Strike

    - **Time to Maturity**: Days until option expiration
      - Short-term: 0-30 days (gamma risk, high theta decay)
      - Medium-term: 30-90 days (balanced risk/reward)
      - Long-term: 90+ days (lower theta, more vega exposure)

    **Trading Insights:**
    - High call volume at specific strikes may indicate resistance levels
    - High put volume may indicate support levels or hedging activity
    - Concentration in short-dated options suggests event-driven trading
    """)

st.markdown("---")
st.info("""
**💡 Pro Tip**: Look for unusual volume spikes at specific strikes or maturities.
These can indicate institutional positioning or upcoming market events.
""")
