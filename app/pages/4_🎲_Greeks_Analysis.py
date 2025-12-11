"""
Greeks Analysis Page
Delta, Gamma, Vega, Theta surfaces and risk metrics
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

from src.models.greeks import PortfolioGreeksAggregator

st.set_page_config(page_title="Greeks Analysis", page_icon="🎲", layout="wide")

st.title("🎲 Greeks Analysis & Risk Dashboard")
st.markdown("### Understanding option sensitivities and portfolio risk")

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
    help="Choose currency to analyze Greeks"
)

# Get data
if f"{selected_currency}_data" not in st.session_state:
    st.error(f"No data available for {selected_currency}")
    st.stop()

data = st.session_state[f"{selected_currency}_data"]
spot_price = data['index_price'].iloc[-1]

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio Greeks",
    "🔷 Delta Analysis",
    "🔶 Gamma Analysis",
    "🔵 Vega Analysis",
    "🔴 Theta Analysis"
])

# === TAB 1: PORTFOLIO GREEKS ===
with tab1:
    st.subheader(f"{selected_currency} Portfolio Greeks Summary")

    # Calculate portfolio Greeks
    portfolio_greeks = PortfolioGreeksAggregator.aggregate_greeks(data, spot_price)

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Delta",
            f"{portfolio_greeks.total_delta:,.2f}",
            delta=f"${portfolio_greeks.delta_dollars:,.0f}",
            help="Portfolio delta exposure (directional risk)"
        )

    with col2:
        st.metric(
            "Total Gamma",
            f"{portfolio_greeks.total_gamma:,.4f}",
            delta=f"${portfolio_greeks.gamma_dollars:,.0f} per 1% move",
            help="Portfolio gamma exposure (delta change risk)"
        )

    with col3:
        st.metric(
            "Total Vega",
            f"{portfolio_greeks.total_vega:,.2f}",
            delta=f"${portfolio_greeks.vega_dollars:,.0f}",
            help="Portfolio vega exposure (volatility risk)"
        )

    with col4:
        st.metric(
            "Total Theta",
            f"{portfolio_greeks.total_theta:,.2f}",
            delta=f"${portfolio_greeks.theta_dollars:,.0f} per day",
            help="Portfolio theta exposure (time decay)"
        )

    # Greeks breakdown by option type
    st.markdown("---")
    st.subheader("Greeks Breakdown by Option Type")

    call_greeks = PortfolioGreeksAggregator.aggregate_greeks(
        data[data['option_type'] == 'call'], spot_price
    )
    put_greeks = PortfolioGreeksAggregator.aggregate_greeks(
        data[data['option_type'] == 'put'], spot_price
    )

    breakdown_data = {
        'Greek': ['Delta', 'Gamma', 'Vega', 'Theta'],
        'Calls': [
            call_greeks.total_delta,
            call_greeks.total_gamma,
            call_greeks.total_vega,
            call_greeks.total_theta
        ],
        'Puts': [
            put_greeks.total_delta,
            put_greeks.total_gamma,
            put_greeks.total_vega,
            put_greeks.total_theta
        ],
        'Net': [
            portfolio_greeks.total_delta,
            portfolio_greeks.total_gamma,
            portfolio_greeks.total_vega,
            portfolio_greeks.total_theta
        ]
    }

    breakdown_df = pd.DataFrame(breakdown_data)

    st.dataframe(
        breakdown_df.style.format({
            'Calls': '{:.4f}',
            'Puts': '{:.4f}',
            'Net': '{:.4f}'
        }).background_gradient(subset=['Net'], cmap='RdYlGn'),
        use_container_width=True
    )

    # Greeks by maturity
    st.markdown("---")
    st.subheader("Greeks Distribution by Maturity")

    maturity_greeks = data.groupby('maturity_date').agg({
        'delta': 'sum',
        'gamma': 'sum',
        'vega': 'sum',
        'theta': 'sum',
        'volume_btc': 'sum'
    }).reset_index()

    maturity_greeks['maturity_date'] = pd.to_datetime(maturity_greeks['maturity_date'])
    maturity_greeks = maturity_greeks.sort_values('maturity_date')

    fig_maturity = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Delta by Maturity", "Gamma by Maturity",
                       "Vega by Maturity", "Theta by Maturity")
    )

    greeks_plots = [
        ('delta', '#636EFA', 1, 1),
        ('gamma', '#EF553B', 1, 2),
        ('vega', '#00CC96', 2, 1),
        ('theta', '#AB63FA', 2, 2)
    ]

    for greek, color, row, col in greeks_plots:
        fig_maturity.add_trace(
            go.Bar(
                x=maturity_greeks['maturity_date'],
                y=maturity_greeks[greek],
                name=greek.capitalize(),
                marker_color=color,
                showlegend=False,
                hovertemplate=(
                    f"<b>{greek.capitalize()}</b><br>" +
                    "Maturity: %{x|%Y-%m-%d}<br>" +
                    f"{greek.capitalize()}: " + "%{y:.4f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=row, col=col
        )

    fig_maturity.update_layout(
        height=700,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_maturity, use_container_width=True)

# === TAB 2: DELTA ANALYSIS ===
with tab2:
    st.subheader("🔷 Delta Analysis")

    # Delta surface
    st.markdown("#### Delta Surface by Strike and Maturity")

    col1, col2 = st.columns(2)
    with col1:
        delta_option_type = st.selectbox(
            "Option Type (Delta)",
            ["call", "put"],
            key="delta_type"
        )

    delta_data = data[data['option_type'] == delta_option_type].copy()

    if not delta_data.empty:
        # 2D heatmap
        pivot_delta = delta_data.pivot_table(
            values='delta',
            index='strike_price',
            columns='maturity_date',
            aggfunc='mean'
        )

        fig_delta = go.Figure(data=go.Heatmap(
            z=pivot_delta.values,
            x=pivot_delta.columns,
            y=pivot_delta.index,
            colorscale='RdBu',
            zmid=0,
            hovertemplate=(
                "<b>Delta</b><br>" +
                "Strike: $%{y:,.0f}<br>" +
                "Maturity: %{x|%Y-%m-%d}<br>" +
                "Delta: %{z:.4f}<br>" +
                "<extra></extra>"
            ),
            colorbar=dict(title="Delta")
        ))

        fig_delta.update_layout(
            title=f"{delta_option_type.capitalize()} Delta Heatmap",
            xaxis_title="Maturity Date",
            yaxis_title="Strike Price (USD)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_delta, use_container_width=True)

        # Delta vs Moneyness
        st.markdown("---")
        st.markdown("#### Delta vs Moneyness")

        fig_delta_moneyness = go.Figure()

        # Group by moneyness bins
        moneyness_bins = pd.cut(delta_data['moneyness'], bins=20)
        delta_by_moneyness = delta_data.groupby(moneyness_bins)['delta'].mean()

        fig_delta_moneyness.add_trace(
            go.Scatter(
                x=[interval.mid for interval in delta_by_moneyness.index],
                y=delta_by_moneyness.values,
                mode='lines+markers',
                name='Delta',
                line=dict(color='#636EFA', width=3),
                marker=dict(size=8),
                hovertemplate=(
                    "<b>Delta</b><br>" +
                    "Moneyness: %{x:.4f}<br>" +
                    "Delta: %{y:.4f}<br>" +
                    "<extra></extra>"
                )
            )
        )

        # Add ATM line
        fig_delta_moneyness.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="yellow",
            annotation_text="ATM",
            annotation_position="top"
        )

        fig_delta_moneyness.update_layout(
            title=f"{delta_option_type.capitalize()} Delta vs Moneyness",
            xaxis_title="Moneyness (Spot/Strike)",
            yaxis_title="Delta",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_delta_moneyness, use_container_width=True)

        with st.expander("📖 Understanding Delta"):
            st.markdown("""
            **Delta Interpretation:**

            - **Call Delta**: 0 to 1
              - Deep ITM calls: Delta ≈ 1 (moves $1 for $1 with underlying)
              - ATM calls: Delta ≈ 0.5
              - Deep OTM calls: Delta ≈ 0

            - **Put Delta**: -1 to 0
              - Deep ITM puts: Delta ≈ -1
              - ATM puts: Delta ≈ -0.5
              - Deep OTM puts: Delta ≈ 0

            **Hedging**: To delta hedge, take opposite position in underlying
            - Positive delta → sell underlying
            - Negative delta → buy underlying
            """)

# === TAB 3: GAMMA ANALYSIS ===
with tab3:
    st.subheader("🔶 Gamma Analysis")

    st.markdown("#### Gamma Surface by Strike and Maturity")

    col1, col2 = st.columns(2)
    with col1:
        gamma_option_type = st.selectbox(
            "Option Type (Gamma)",
            ["call", "put", "both"],
            key="gamma_type"
        )

    if gamma_option_type == "both":
        gamma_data = data.copy()
    else:
        gamma_data = data[data['option_type'] == gamma_option_type].copy()

    if not gamma_data.empty:
        # 2D heatmap
        pivot_gamma = gamma_data.pivot_table(
            values='gamma',
            index='strike_price',
            columns='maturity_date',
            aggfunc='mean'
        )

        fig_gamma = go.Figure(data=go.Heatmap(
            z=pivot_gamma.values,
            x=pivot_gamma.columns,
            y=pivot_gamma.index,
            colorscale='Hot',
            hovertemplate=(
                "<b>Gamma</b><br>" +
                "Strike: $%{y:,.0f}<br>" +
                "Maturity: %{x|%Y-%m-%d}<br>" +
                "Gamma: %{z:.6f}<br>" +
                "<extra></extra>"
            ),
            colorbar=dict(title="Gamma")
        ))

        fig_gamma.update_layout(
            title="Gamma Heatmap",
            xaxis_title="Maturity Date",
            yaxis_title="Strike Price (USD)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_gamma, use_container_width=True)

        # Gamma by strike (near ATM)
        st.markdown("---")
        st.markdown("#### Gamma Profile Near ATM")

        atm_range = (spot_price * 0.8, spot_price * 1.2)
        atm_gamma = gamma_data[
            (gamma_data['strike_price'] >= atm_range[0]) &
            (gamma_data['strike_price'] <= atm_range[1])
        ].copy()

        if not atm_gamma.empty:
            gamma_by_strike = atm_gamma.groupby('strike_price')['gamma'].sum().reset_index()

            fig_gamma_strike = go.Figure()

            fig_gamma_strike.add_trace(
                go.Bar(
                    x=gamma_by_strike['strike_price'],
                    y=gamma_by_strike['gamma'],
                    marker_color='#EF553B',
                    hovertemplate=(
                        "<b>Gamma</b><br>" +
                        "Strike: $%{x:,.0f}<br>" +
                        "Total Gamma: %{y:.6f}<br>" +
                        "<extra></extra>"
                    )
                )
            )

            fig_gamma_strike.add_vline(
                x=spot_price,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Spot: ${spot_price:,.0f}",
                annotation_position="top"
            )

            fig_gamma_strike.update_layout(
                title="Total Gamma by Strike (±20% from spot)",
                xaxis_title="Strike Price (USD)",
                yaxis_title="Total Gamma",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_gamma_strike, use_container_width=True)

        with st.expander("📖 Understanding Gamma"):
            st.markdown("""
            **Gamma Interpretation:**

            - **Gamma**: Rate of change of delta with respect to underlying price
            - **Always positive** for both calls and puts (from buyer perspective)
            - **Highest at ATM**, decreases as option moves ITM or OTM
            - **Increases as expiration approaches** (short-dated options more sensitive)

            **Risk Implications:**
            - High gamma → delta changes rapidly → frequent rehedging needed
            - Short gamma (selling options) → unlimited risk in fast markets
            - Long gamma (buying options) → profit from large moves in either direction

            **Gamma Scalping**: Trading strategy exploiting high gamma near ATM
            """)

# === TAB 4: VEGA ANALYSIS ===
with tab4:
    st.subheader("🔵 Vega Analysis")

    st.markdown("#### Vega Surface by Strike and Maturity")

    col1, col2 = st.columns(2)
    with col1:
        vega_option_type = st.selectbox(
            "Option Type (Vega)",
            ["call", "put", "both"],
            key="vega_type"
        )

    if vega_option_type == "both":
        vega_data = data.copy()
    else:
        vega_data = data[data['option_type'] == vega_option_type].copy()

    if not vega_data.empty:
        # 2D heatmap
        pivot_vega = vega_data.pivot_table(
            values='vega',
            index='strike_price',
            columns='maturity_date',
            aggfunc='mean'
        )

        fig_vega = go.Figure(data=go.Heatmap(
            z=pivot_vega.values,
            x=pivot_vega.columns,
            y=pivot_vega.index,
            colorscale='Teal',
            hovertemplate=(
                "<b>Vega</b><br>" +
                "Strike: $%{y:,.0f}<br>" +
                "Maturity: %{x|%Y-%m-%d}<br>" +
                "Vega: %{z:.4f}<br>" +
                "<extra></extra>"
            ),
            colorbar=dict(title="Vega")
        ))

        fig_vega.update_layout(
            title="Vega Heatmap",
            xaxis_title="Maturity Date",
            yaxis_title="Strike Price (USD)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_vega, use_container_width=True)

        # Vega vs IV
        st.markdown("---")
        st.markdown("#### Vega vs Implied Volatility")

        fig_vega_iv = go.Figure()

        fig_vega_iv.add_trace(
            go.Scatter(
                x=vega_data['iv'] * 100,
                y=vega_data['vega'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=vega_data['time_to_maturity'],
                    colorscale='Viridis',
                    colorbar=dict(title="TTM (days)"),
                    showscale=True
                ),
                hovertemplate=(
                    "<b>Vega vs IV</b><br>" +
                    "IV: %{x:.2f}%<br>" +
                    "Vega: %{y:.4f}<br>" +
                    "<extra></extra>"
                )
            )
        )

        fig_vega_iv.update_layout(
            title="Vega vs Implied Volatility (colored by TTM)",
            xaxis_title="Implied Volatility (%)",
            yaxis_title="Vega",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_vega_iv, use_container_width=True)

        with st.expander("📖 Understanding Vega"):
            st.markdown("""
            **Vega Interpretation:**

            - **Vega**: Sensitivity to changes in implied volatility (per 1% change)
            - **Always positive** for both calls and puts (from buyer perspective)
            - **Highest for ATM options**
            - **Increases with time to maturity** (longer-dated options more sensitive)

            **Trading Strategies:**
            - Long vega (buy options) → profit from vol increase
            - Short vega (sell options) → profit from vol decrease
            - Vega hedging: Offset volatility risk across different strikes/maturities

            **Important**: Vega exposure is critical in crypto markets due to high volatility
            """)

# === TAB 5: THETA ANALYSIS ===
with tab5:
    st.subheader("🔴 Theta Analysis")

    st.markdown("#### Theta (Time Decay) Surface")

    col1, col2 = st.columns(2)
    with col1:
        theta_option_type = st.selectbox(
            "Option Type (Theta)",
            ["call", "put"],
            key="theta_type"
        )

    theta_data = data[data['option_type'] == theta_option_type].copy()

    if not theta_data.empty:
        # 2D heatmap
        pivot_theta = theta_data.pivot_table(
            values='theta',
            index='strike_price',
            columns='maturity_date',
            aggfunc='mean'
        )

        fig_theta = go.Figure(data=go.Heatmap(
            z=pivot_theta.values,
            x=pivot_theta.columns,
            y=pivot_theta.index,
            colorscale='Reds',
            reversescale=True,
            hovertemplate=(
                "<b>Theta</b><br>" +
                "Strike: $%{y:,.0f}<br>" +
                "Maturity: %{x|%Y-%m-%d}<br>" +
                "Theta: %{z:.6f}<br>" +
                "<extra></extra>"
            ),
            colorbar=dict(title="Theta (per day)")
        ))

        fig_theta.update_layout(
            title=f"{theta_option_type.capitalize()} Theta Heatmap",
            xaxis_title="Maturity Date",
            yaxis_title="Strike Price (USD)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_theta, use_container_width=True)

        # Theta decay curve
        st.markdown("---")
        st.markdown("#### Theta Decay vs Time to Maturity")

        theta_by_ttm = theta_data.groupby('time_to_maturity')['theta'].mean().reset_index()
        theta_by_ttm = theta_by_ttm.sort_values('time_to_maturity')

        fig_theta_curve = go.Figure()

        fig_theta_curve.add_trace(
            go.Scatter(
                x=theta_by_ttm['time_to_maturity'],
                y=theta_by_ttm['theta'],
                mode='lines+markers',
                name='Theta',
                line=dict(color='#EF553B', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                hovertemplate=(
                    "<b>Theta Decay</b><br>" +
                    "TTM: %{x:.1f} days<br>" +
                    "Theta: %{y:.6f}<br>" +
                    "<extra></extra>"
                )
            )
        )

        fig_theta_curve.update_layout(
            title="Average Theta Decay Curve",
            xaxis_title="Time to Maturity (Days)",
            yaxis_title="Theta (per day)",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_theta_curve, use_container_width=True)

        with st.expander("📖 Understanding Theta"):
            st.markdown("""
            **Theta Interpretation:**

            - **Theta**: Rate of time decay (option value lost per day)
            - **Negative for long options** (value decreases over time)
            - **Positive for short options** (profit from time decay)
            - **Accelerates as expiration approaches** (non-linear decay)

            **Key Characteristics:**
            - **Highest for ATM options**
            - **Increases dramatically in final week before expiration**
            - **Lower for deep ITM or OTM options**

            **Trading Implications:**
            - Option buyers fight theta decay
            - Option sellers profit from theta decay
            - Weekend effect: 3 days of decay over weekend
            - Theta gang: Selling options to capture time decay
            """)

st.markdown("---")
st.info("""
**💡 Pro Tip**: Monitor portfolio Greeks daily. Large concentrations in any Greek
can lead to unexpected risk during market moves or volatility changes.
""")
