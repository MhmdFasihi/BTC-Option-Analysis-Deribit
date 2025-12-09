import numpy as np
import pandas as pd
import requests
from datetime import datetime as dt
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os
import json

warnings.filterwarnings('ignore')

def datetime_to_timestamp(datetime_obj: dt) -> int:
    """Convert datetime to millisecond timestamp"""
    return int(dt.timestamp(datetime_obj) * 1000)

class OptionsDataFetcher:
    def __init__(self, currencies: Union[List[str], str], start_date: date, end_date: date, 
                 cache_dir: str = "cache", max_workers: int = 5):
        self.currencies = [currencies.upper()] if isinstance(currencies, str) else [c.upper() for c in currencies]
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.session = requests.Session()
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        for currency in self.currencies:
            if currency not in ['BTC', 'ETH']:
                raise ValueError(f"Currency {currency} is not supported. Choose from BTC, ETH")
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
        if (self.end_date - self.start_date).days > 365:
            raise ValueError("Date range cannot exceed 365 days")
            
    def _get_cache_filename(self, currency: str) -> Path:
        return self.cache_dir / f"{currency}_options_{self.start_date}_{self.end_date}.csv"
    
    def _fetch_trades_chunk(self, currency: str, start_ts: int, end_ts: int) -> List[Dict]:
        params = {
            "currency": currency,
            "kind": "option",
            "count": 10000,
            "include_old": True,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts
        }
        
        url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'
        
        for attempt in range(5):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if "result" in data and "trades" in data["result"]:
                    return data["result"]["trades"]
                return []
            except Exception as e:
                if attempt == 4:
                    print(f"Failed to get data for {currency} from {start_ts} to {end_ts}: {e}")
                    return []
                time.sleep(0.5 * (2 ** attempt))
        return []
    
    def _parse_instrument_name(self, name: str) -> Tuple[str, float, str]:
        try:
            parts = name.split('-')
            maturity = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            return maturity, strike, option_type
        except Exception as e:
            print(f"Error parsing instrument name {name}: {e}")
            return None, None, None
    
    def _process_chunk(self, chunk: List[Dict]) -> pd.DataFrame:
        if not chunk:
            return pd.DataFrame()
            
        df = pd.DataFrame(chunk)
        
        # Parse instrument names
        instrument_details = [self._parse_instrument_name(name) for name in df['instrument_name']]
        df['maturity_date'] = [details[0] for details in instrument_details]
        df['strike_price'] = [details[1] for details in instrument_details]
        df['option_type'] = [details[2] for details in instrument_details]
        
        # Convert timestamp and maturity date
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['maturity_date'] = pd.to_datetime(df['maturity_date'], format='%d%b%y')
        
        # Create derived columns
        df['date'] = df['date_time'].dt.date
        df['time_to_maturity'] = (df['maturity_date'] - df['date_time']).dt.total_seconds() / 86400
        df['moneyness'] = df['index_price'] / df['strike_price']
        df['iv'] = df['iv'] / 100
        df['is_call'] = df['option_type'] == 'call'
        
        # Calculate volume metrics
        df['volume_btc'] = df['price'] * df['contracts']
        df['volume_usd'] = df['volume_btc'] * df['index_price']
        
        return df
    
    def fetch_data(self, currency: str) -> pd.DataFrame:
        cache_file = self._get_cache_filename(currency)
        
        if cache_file.exists():
            print(f"Loading {currency} data from cache...")
            return pd.read_csv(cache_file, parse_dates=['date_time', 'maturity_date'])
            
        print(f"Fetching {currency} data from API...")
        
        # Create date chunks for parallel processing
        date_chunks = []
        current_date = self.start_date
        while current_date < self.end_date:
            next_date = min(current_date + timedelta(days=1), self.end_date)
            date_chunks.append((
                datetime_to_timestamp(dt.combine(current_date, dt.min.time())),
                datetime_to_timestamp(dt.combine(next_date, dt.max.time()))
            ))
            current_date = next_date
            
        all_trades = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._fetch_trades_chunk, currency, start_ts, end_ts)
                for start_ts, end_ts in date_chunks
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {currency} data"):
                trades = future.result()
                if trades:
                    all_trades.extend(trades)
                    
        if not all_trades:
            print(f"No data retrieved for {currency}")
            return pd.DataFrame()
            
        df = self._process_chunk(all_trades)
        df.to_csv(cache_file, index=False)
        return df
    
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        results = {}
        for currency in self.currencies:
            results[currency] = self.fetch_data(currency)
        return results

class OptionsAnalyzer:
    def __init__(self, data: pd.DataFrame, currency: str, rate: float = 0.03, output_path: Optional[str] = None):
        self.data = data
        self.currency = currency
        self.rate = rate
        self.output_path = output_path or os.getcwd()
        self.results_dir = Path(self.output_path) / "results" / currency
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def _calculate_greeks(self, row: pd.Series) -> Tuple[float, float, float, float]:
        S = row['index_price']
        K = row['strike_price']
        T = max(row['time_to_maturity'] / 365, 1e-6)
        sigma = row['iv']
        r = self.rate
        is_call = row['option_type'] == 'call'

        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta calculation
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)

        # Gamma calculation
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega calculation (for 1% change in vol)
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01
        
        # Theta calculation (for 1 day)
        theta_common = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))
        if is_call:
            theta = theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert theta to daily
        theta = theta / 365
        
        return delta, gamma, vega, theta
        
    def calculate_greeks(self) -> pd.DataFrame:
        if self.data.empty:
            return self.data
            
        print(f"Calculating Greeks for {self.currency}...")
        
        # Apply Greek calculations to each row
        greeks = self.data.apply(self._calculate_greeks, axis=1, result_type='expand')
        greeks.columns = ['delta', 'gamma', 'vega', 'theta']
        
        # Combine with original data
        enhanced_data = pd.concat([self.data, greeks], axis=1)
        
        # Calculate call/put ratio by date
        daily_cp = enhanced_data.groupby('date').agg({
            'is_call': ['sum', 'size']
        })
        
        daily_cp.columns = ['calls', 'total']
        daily_cp['puts'] = daily_cp['total'] - daily_cp['calls']
        daily_cp['cp_ratio'] = daily_cp['calls'] / daily_cp['puts']
        daily_cp = daily_cp.reset_index()
        
        # Merge back to main dataframe
        enhanced_data = enhanced_data.merge(daily_cp[['date', 'cp_ratio']], on='date', how='left')
        
        return enhanced_data
        
    def analyze_volumes(self) -> Dict:
        if self.data.empty:
            return {'calls': {}, 'puts': {}}
            
        calls = self.data[self.data['option_type'] == 'call']
        puts = self.data[self.data['option_type'] == 'put']
        
        analysis = {
            'calls': {
                'count': len(calls),
                'volume_btc': calls['volume_btc'].sum(),
                'volume_usd': calls['volume_usd'].sum(),
                'avg_size_btc': calls['volume_btc'].mean(),
                'avg_size_usd': calls['volume_usd'].mean(),
                'num_trades': len(calls),
                'unique_strikes': len(calls['strike_price'].unique()),
                'avg_iv': calls['iv'].mean(),
                'avg_delta': calls['delta'].mean() if 'delta' in calls.columns else None
            },
            'puts': {
                'count': len(puts),
                'volume_btc': puts['volume_btc'].sum(),
                'volume_usd': puts['volume_usd'].sum(),
                'avg_size_btc': puts['volume_btc'].mean(),
                'avg_size_usd': puts['volume_usd'].mean(),
                'num_trades': len(puts),
                'unique_strikes': len(puts['strike_price'].unique()),
                'avg_iv': puts['iv'].mean(),
                'avg_delta': puts['delta'].mean() if 'delta' in puts.columns else None
            }
        }
        
        # Add overall statistics
        analysis['total'] = {
            'count': len(self.data),
            'volume_btc': self.data['volume_btc'].sum(),
            'volume_usd': self.data['volume_usd'].sum(),
            'put_call_ratio': puts['volume_usd'].sum() / max(calls['volume_usd'].sum(), 1),
            'unique_strikes': len(self.data['strike_price'].unique()),
            'unique_maturities': len(self.data['maturity_date'].unique())
        }
        
        return analysis
        
    def generate_summary_report(self) -> str:
        if self.data.empty:
            return f"No data available for {self.currency}"
            
        volumes = self.analyze_volumes()
        
        # Get latest index price
        current_price = self.data['index_price'].iloc[-1]
        
        report = f"""
{self.currency} Options Analysis Report
========================================
Analysis Period: {self.data['date_time'].min().date()} to {self.data['date_time'].max().date()}
Current {self.currency} Price: ${current_price:,.2f}

1. Trading Volume Summary:
-------------------------
Call Options:
- Number of trades: {volumes['calls']['count']:,}
- Total volume ({self.currency}): {volumes['calls']['volume_btc']:,.4f}
- Total volume (USD): ${volumes['calls']['volume_usd']:,.2f}
- Average trade size ({self.currency}): {volumes['calls']['avg_size_btc']:.4f}
- Average IV: {volumes['calls']['avg_iv']:.2%}
- Unique strike prices: {volumes['calls']['unique_strikes']}

Put Options:
- Number of trades: {volumes['puts']['count']:,}
- Total volume ({self.currency}): {volumes['puts']['volume_btc']:,.4f}
- Total volume (USD): ${volumes['puts']['volume_usd']:,.2f}
- Average trade size ({self.currency}): {volumes['puts']['avg_size_btc']:.4f}
- Average IV: {volumes['puts']['avg_iv']:.2%}
- Unique strike prices: {volumes['puts']['unique_strikes']}

2. Market Statistics:
--------------------
Put/Call Ratio (by count): {volumes['puts']['count'] / max(volumes['calls']['count'], 1):.2f}
Put/Call Ratio (by volume): {volumes['puts']['volume_btc'] / max(volumes['calls']['volume_btc'], 1):.2f}
"""

        # Most active strikes
        if not self.data[self.data['option_type'] == 'call'].empty:
            report += f"Most Active Strike (Calls): {self.data[self.data['option_type'] == 'call']['strike_price'].value_counts().idxmax():,.0f}\n"
        if not self.data[self.data['option_type'] == 'put'].empty:
            report += f"Most Active Strike (Puts): {self.data[self.data['option_type'] == 'put']['strike_price'].value_counts().idxmax():,.0f}\n"
            
        # Most traded maturities
        if not self.data.empty:
            most_traded_maturity = self.data.groupby('maturity_date')['volume_usd'].sum().idxmax()
            report += f"Most Traded Maturity: {most_traded_maturity.strftime('%Y-%m-%d')}\n\n"
            
        # Daily call/put ratio
        report += "3. Daily Call/Put Ratio:\n------------------------\n"
        daily_cp_ratio = self.data.groupby('date').agg({
            'is_call': ['sum', 'size']
        }).reset_index()
        daily_cp_ratio.columns = ['date', 'calls', 'total']
        daily_cp_ratio['puts'] = daily_cp_ratio['total'] - daily_cp_ratio['calls']
        daily_cp_ratio['cp_ratio'] = daily_cp_ratio['calls'] / daily_cp_ratio['puts']
        
        for _, row in daily_cp_ratio.iterrows():
            report += f"{row['date']}: {row['cp_ratio']:.2f} (Calls: {row['calls']}, Puts: {row['puts']})\n"
            
        # IV statistics
        report += "\n4. Implied Volatility Statistics:\n-------------------------------\n"
        for option_type in ['call', 'put']:
            type_data = self.data[self.data['option_type'] == option_type]
            if not type_data.empty:
                report += f"{option_type.capitalize()} Options IV Summary:\n"
                report += f"- Mean IV: {type_data['iv'].mean():.2%}\n"
                report += f"- Median IV: {type_data['iv'].median():.2%}\n"
                report += f"- Min IV: {type_data['iv'].min():.2%}\n"
                report += f"- Max IV: {type_data['iv'].max():.2%}\n\n"
                
        return report

class OptionsVisualizer:
    def __init__(self, data: pd.DataFrame, currency: str, results_dir: Path):
        self.data = data
        self.currency = currency
        self.results_dir = results_dir
        
    def plot_volatility_surface(self) -> go.Figure:
        if self.data.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No data available for {self.currency} volatility surface")
            return fig
            
        fig = go.Figure()
        
        # Add surface plot
        fig.add_trace(go.Scatter3d(
            x=self.data['moneyness'],
            y=self.data['time_to_maturity'],
            z=self.data['iv'],
            mode='markers',
            marker=dict(
                size=4,
                color=self.data['iv'],
                colorscale='Viridis',
                opacity=0.8
            ),
            name='IV Surface'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{self.currency} Volatility Surface",
            scene=dict(
                xaxis_title="Moneyness (Spot/Strike)",
                yaxis_title="Time to Maturity (days)",
                zaxis_title="Implied Volatility"
            ),
            width=1000,
            height=800
        )
        
        return fig
        
    def plot_greeks_3d(self) -> Dict[str, go.Figure]:
        if self.data.empty or 'delta' not in self.data.columns:
            return {}
            
        figures = {}
        
        for greek in ['delta', 'gamma', 'vega', 'theta']:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter3d(
                x=self.data['time_to_maturity'],
                y=self.data['moneyness'],
                z=self.data[greek],
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.data[greek],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name=f'{greek.capitalize()} Surface'
            ))
            
            fig.update_layout(
                title=f"{self.currency} {greek.capitalize()} Surface",
                scene=dict(
                    xaxis_title="Time to Maturity (days)",
                    yaxis_title="Moneyness (Spot/Strike)",
                    zaxis_title=greek.capitalize()
                ),
                width=1000,
                height=800
            )
            
            figures[greek] = fig
            
        return figures
        
    def plot_greeks_vs_moneyness(self) -> go.Figure:
        if self.data.empty or 'delta' not in self.data.columns:
            fig = go.Figure()
            fig.update_layout(title=f"No data available for {self.currency} Greeks")
            return fig
            
        fig = go.Figure()
        
        # Plot Greeks vs moneyness
        for greek in ['delta', 'gamma', 'vega', 'theta']:
            fig.add_trace(go.Scatter(
                x=self.data['moneyness'],
                y=self.data[greek],
                mode='markers',
                name=greek.capitalize(),
                marker=dict(size=4)
            ))
            
        fig.update_layout(
            title=f"{self.currency} Option Greeks vs Moneyness",
            xaxis_title="Moneyness (Spot/Strike)",
            yaxis_title="Greek Value",
            width=1000,
            height=600
        )
        
        return fig
        
    def plot_greeks_vs_iv(self) -> go.Figure:
        if self.data.empty or 'delta' not in self.data.columns:
            fig = go.Figure()
            fig.update_layout(title=f"No data available for {self.currency} Greeks vs IV")
            return fig
            
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Delta vs IV', 'Gamma vs IV',
                                         'Vega vs IV', 'Theta vs IV'))
        
        row_col = [(1,1), (1,2), (2,1), (2,2)]
        for greek, (row, col) in zip(['delta', 'gamma', 'vega', 'theta'], row_col):
            fig.add_trace(
                go.Scatter(
                    x=self.data['iv'],
                    y=self.data[greek],
                    mode='markers',
                    name=greek.capitalize(),
                    marker=dict(size=4)
                ),
                row=row, col=col
            )
            
        fig.update_layout(
            title=f"{self.currency} Option Greeks vs Implied Volatility",
            showlegend=True,
            width=1200,
            height=800
        )
        
        # Update all subplot axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Implied Volatility", row=i, col=j)
                fig.update_yaxes(title_text="Greek Value", row=i, col=j)
                
        return fig
        
    def plot_cp_ratio(self) -> go.Figure:
        if self.data.empty or 'cp_ratio' not in self.data.columns:
            fig = go.Figure()
            fig.update_layout(title=f"No data available for {self.currency} Call/Put ratio")
            return fig
            
        daily_data = self.data.groupby('date_time').first().reset_index()
        
        fig = go.Figure()
        
        # Plot call/put ratio
        fig.add_trace(go.Scatter(
            x=daily_data['date_time'],
            y=daily_data['cp_ratio'],
            mode='lines+markers',
            name='Call/Put Ratio',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        
        # Add reference line at 1.0
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                    annotation_text="Neutral Level")
        
        fig.update_layout(
            title=f"{self.currency} Daily Call/Put Ratio",
            xaxis_title="Date",
            yaxis_title="Call/Put Ratio",
            width=1000,
            height=600,
            showlegend=True
        )
        
        return fig
        
    def plot_option_distributions(self) -> Dict[str, go.Figure]:
        if self.data.empty:
            return {}
            
        calls = self.data[self.data['option_type'] == 'call']
        puts = self.data[self.data['option_type'] == 'put']
        current_price = self.data['index_price'].iloc[-1]
        figures = {}
        
        # Helper function for statistical calculations
        def get_stats(data):
            if data.empty:
                return {}
            return {
                'mean': data['strike_price'].mean(),
                'median': data['strike_price'].median(),
                'std': data['strike_price'].std(),
                'volume_weighted_avg': (data['strike_price'] * data['volume_usd']).sum() / max(data['volume_usd'].sum(), 1),
                'total_volume': data['volume_usd'].sum(),
                'contract_count': len(data),
                'iv_mean': data['iv'].mean(),
                'delta_mean': data['delta'].mean() if 'delta' in data.columns else None
            }
            
        call_stats = get_stats(calls)
        put_stats = get_stats(puts)
        
        # 1. Enhanced Strike Price Distribution
        if not calls.empty or not puts.empty:
            fig_strike = go.Figure()
            
            # Add violin plots with statistical information
            for opt_type, data, stats, color in [
                ('Calls', calls, call_stats, 'blue'),
                ('Puts', puts, put_stats, 'red')
            ]:
                if data.empty:
                    continue
                    
                # Add violin plot
                fig_strike.add_trace(go.Violin(
                    x=data['strike_price'],
                    name=opt_type,
                    side='positive' if opt_type == 'Calls' else 'negative',
                    line_color=color,
                    fillcolor=f'rgba{(0, 0, 255, 0.2) if color=="blue" else (255, 0, 0, 0.2)}',
                    opacity=0.6,
                    points=False,
                    box=dict(visible=True),
                    meanline=dict(visible=True),
                    hoverinfo='skip'
                ))
                
                # Add scatter points for trades
                fig_strike.add_trace(go.Scatter(
                    x=data['strike_price'],
                    y=[0] * len(data),
                    mode='markers',
                    name=f'{opt_type} Trades',
                    marker=dict(
                        color=color,
                        size=6,
                        opacity=0.4
                    ),
                    hovertemplate=(
                        f"<b>{opt_type}</b><br>" +
                        "Strike: $%{x:,.0f}<br>" +
                        "Volume: $%{customdata[0]:,.2f}<br>" +
                        "IV: %{customdata[1]:.1%}<br>" +
                        ("Delta: %{customdata[2]:.3f}<br>" if 'delta' in data.columns else "") +
                        "<extra></extra>"
                    ),
                    customdata=data[['volume_usd', 'iv'] + (['delta'] if 'delta' in data.columns else [])]
                ))
                
                # Add statistical annotations
                if stats:
                    stats_text = (
                        f"<b>{opt_type} Statistics</b><br>" +
                        f"Mean Strike: ${stats['mean']:,.0f}<br>" +
                        f"Median Strike: ${stats['median']:,.0f}<br>" +
                        f"Std Dev: ${stats['std']:,.0f}<br>" +
                        f"Volume Weighted Avg: ${stats['volume_weighted_avg']:,.0f}<br>" +
                        f"Total Volume: ${stats['total_volume']:,.2f}<br>" +
                        f"Contracts: {stats['contract_count']:,}<br>" +
                        f"Avg IV: {stats['iv_mean']:.1%}"
                    )
                    
                    if stats['delta_mean'] is not None:
                        stats_text += f"<br>Avg Delta: {stats['delta_mean']:.3f}"
                        
                    fig_strike.add_annotation(
                        x=current_price * (1.2 if opt_type == 'Calls' else 0.8),
                        y=1 if opt_type == 'Calls' else -1,
                        text=stats_text,
                        showarrow=False,
                        font=dict(size=10, color=color),
                        align='left',
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor=color,
                        borderwidth=1
                    )
                    
            # Add current price line
            fig_strike.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="green",
                annotation=dict(
                    text=f"Current Price: ${current_price:,.0f}",
                    font=dict(size=12),
                    bordercolor="green",
                    borderwidth=1,
                    bgcolor="white"
                )
            )
            
            fig_strike.update_layout(
                title=dict(
                    text=f"{self.currency} Options Distribution by Strike Price",
                    font=dict(size=24),
                    x=0.5
                ),
                xaxis=dict(
                    title=dict(
                        text="Strike Price (USD)",
                        font=dict(size=16)
                    ),
                    tickformat="$,.0f",
                    gridcolor='lightgray',
                    showgrid=True,
                    range=[current_price * 0.5, current_price * 2]
                ),
                yaxis=dict(
                    title=dict(
                        text="Distribution",
                        font=dict(size=16)
                    ),
                    gridcolor='lightgray',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                ),
                showlegend=True,
                legend=dict(
                    x=1.15,
                    y=0.95,
                    bordercolor="black",
                    borderwidth=1,
                    bgcolor="white"
                ),
                hovermode='closest',
                plot_bgcolor='white',
                width=1400,
                height=800,
                margin=dict(t=100, b=50, l=50, r=200)
            )
            figures['strike'] = fig_strike
            
        # 2. Maturity Distribution
        if not calls.empty or not puts.empty:
            fig_maturity = go.Figure()
            
            # Calculate volume-weighted stats by maturity
            maturity_stats = {}
            for opt_type, data in [('Calls', calls), ('Puts', puts)]:
                if data.empty:
                    continue
                    
                stats = data.groupby('maturity_date').agg({
                    'contracts': 'sum',
                    'volume_usd': 'sum',
                    'iv': 'mean'
                })
                
                if 'delta' in data.columns:
                    stats['delta'] = data.groupby('maturity_date')['delta'].mean()
                    
                stats = stats.reset_index()
                maturity_stats[opt_type] = stats
                
                customdata_cols = ['volume_usd', 'iv']
                if 'delta' in stats.columns:
                    customdata_cols.append('delta')
                    
                fig_maturity.add_trace(go.Bar(
                    x=stats['maturity_date'],
                    y=stats['contracts'],
                    name=opt_type,
                    marker_color='blue' if opt_type == 'Calls' else 'red',
                    opacity=0.7,
                    hovertemplate=(
                        f"<b>{opt_type}</b><br>" +
                        "Maturity: %{x|%Y-%m-%d}<br>" +
                        "Contracts: %{y:,}<br>" +
                        "Volume: $%{customdata[0]:,.2f}<br>" +
                        "Avg IV: %{customdata[1]:.1%}<br>" +
                        ("Avg Delta: %{customdata[2]:.3f}<br>" if 'delta' in stats.columns else "") +
                        "<extra></extra>"
                    ),
                    customdata=stats[customdata_cols]
                ))
            
            fig_maturity.update_layout(
                title=dict(text=f"{self.currency} Options Distribution by Maturity", x=0.5),
                xaxis_title="Maturity Date",
                yaxis_title="Number of Contracts",
                barmode='group',
                width=1200,
                height=600,
                showlegend=True,
                legend=dict(x=1.02, y=0.98),
                hovermode='x unified',
                plot_bgcolor='white'
            )
            figures['maturity'] = fig_maturity
            
        # 3. Volume Distribution
        if not calls.empty or not puts.empty:
            fig_volume = go.Figure()
            
            # Volume aggregation with all metrics
            for opt_type, data, color in [('Calls', calls, 'blue'), ('Puts', puts, 'red')]:
                if data.empty:
                    continue
                    
                volume_stats = data.groupby('strike_price').agg({
                    'volume_usd': 'sum',
                    'contracts': 'sum',
                    'iv': 'mean'
                })
                
                if 'delta' in data.columns:
                    volume_stats['delta'] = data.groupby('strike_price')['delta'].mean()
                    
                volume_stats = volume_stats.reset_index()
                
                customdata_cols = ['contracts', 'iv']
                if 'delta' in volume_stats.columns:
                    customdata_cols.append('delta')
                
                fig_volume.add_trace(go.Scatter(
                    x=volume_stats['strike_price'],
                    y=volume_stats['volume_usd'],
                    name=opt_type,
                    mode='markers',
                    marker=dict(size=10, color=color, opacity=0.6),
                    hovertemplate=(
                        f"<b>{opt_type}</b><br>" +
                        "Strike: $%{x:,.0f}<br>" +
                        "Volume: $%{y:,.2f}<br>" +
                        "Contracts: %{customdata[0]:,}<br>" +
                        "Avg IV: %{customdata[1]:.1%}<br>" +
                        ("Avg Delta: %{customdata[2]:.3f}<br>" if 'delta' in volume_stats.columns else "") +
                        "<extra></extra>"
                    ),
                    customdata=volume_stats[customdata_cols]
                ))
                
            fig_volume.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Current Price: ${current_price:,.0f}"
            )
            
            fig_volume.update_layout(
                title=dict(text=f"{self.currency} Option Volume Distribution", x=0.5),
                xaxis_title="Strike Price (USD)",
                yaxis_title="Volume (USD)",
                width=1200,
                height=800,
                showlegend=True,
                legend=dict(x=1.02, y=0.98),
                hovermode='closest',
                plot_bgcolor='white',
                yaxis_type='log'
            )
            figures['volume'] = fig_volume
            
        # 4. Time Series Volume
        if not calls.empty or not puts.empty:
            fig_timeseries = go.Figure()
            
            for opt_type, data, color in [('Calls', calls, 'blue'), ('Puts', puts, 'red')]:
                if data.empty:
                    continue
                    
                daily_stats = data.groupby(data['date_time'].dt.date).agg({
                    'volume_usd': 'sum',
                    'contracts': 'sum',
                    'iv': 'mean'
                })
                
                if 'delta' in data.columns:
                    daily_stats['delta'] = data.groupby(data['date_time'].dt.date)['delta'].mean()
                    
                daily_stats = daily_stats.reset_index()
                
                customdata_cols = ['contracts', 'iv']
                if 'delta' in daily_stats.columns:
                    customdata_cols.append('delta')
                
                fig_timeseries.add_trace(go.Bar(
                    x=daily_stats['date_time'],
                    y=daily_stats['volume_usd'],
                    name=f"{opt_type} Volume",
                    marker_color=color,
                    opacity=0.6,
                    hovertemplate=(
                        f"<b>{opt_type}</b><br>" +
                        "Date: %{x|%Y-%m-%d}<br>" +
                        "Volume: $%{y:,.2f}<br>" +
                        "Contracts: %{customdata[0]:,}<br>" +
                        "Avg IV: %{customdata[1]:.1%}<br>" +
                        ("Avg Delta: %{customdata[2]:.3f}<br>" if 'delta' in daily_stats.columns else "") +
                        "<extra></extra>"
                    ),
                    customdata=daily_stats[customdata_cols]
                ))
                
            fig_timeseries.update_layout(
                title=dict(text=f"{self.currency} Daily Option Volume", x=0.5),
                xaxis_title="Date",
                yaxis_title="Volume (USD)",
                barmode='group',
                width=1200,
                height=600,
                showlegend=True,
                legend=dict(x=1.02, y=0.98),
                hovermode='x unified',
                plot_bgcolor='white'
            )
            figures['timeseries'] = fig_timeseries
            
        # 5. Option Price Over Time
        figures['price_time'] = go.Figure()
        figures['price_time'].add_trace(go.Scatter(
            x=self.data.sort_values('date_time')['date_time'],
            y=self.data.sort_values('date_time')['price'],
            mode='lines',
            name='Option Price'
        ))
        figures['price_time'].update_layout(
            title='Option Price Over Time',
            xaxis_title='Date',
            yaxis_title='Option Price',
            height=800
        )
        
        # 6. Strike Price Distribution by Maturity Date (Box Plot)
        figures['strike_box'] = go.Figure()
        dates = sorted(self.data['maturity_date'].unique())
        
        for date in dates:
            date_data = self.data[self.data['maturity_date'] == date]['strike_price']
            figures['strike_box'].add_trace(go.Box(
                y=date_data,
                name=date.strftime('%Y-%m-%d'),
                boxpoints='outliers'
            ))
            
        figures['strike_box'].update_layout(
            title='Strike Price Distribution by Maturity Date',
            xaxis_title='Maturity Date',
            yaxis_title='Strike Price',
            xaxis_tickangle=45,
            height=800
        )
        
        # 7. IV Distribution with 10% Bins
        iv_bins = pd.cut(self.data['iv'], bins=np.arange(0, 1.1, 0.1))
        iv_counts = iv_bins.value_counts().sort_index()
        
        figures['iv_distribution'] = go.Figure()
        figures['iv_distribution'].add_trace(go.Bar(
            x=[f"{int(interval.left*100)}-{int(interval.right*100)}%" for interval in iv_counts.index],
            y=iv_counts.values,
            name='IV Distribution'
        ))
        
        figures['iv_distribution'].update_layout(
            title='Distribution of Implied Volatility (IV)',
            xaxis_title='Implied Volatility Range',
            yaxis_title='Frequency',
            height=800
        )
            
        return figures
        
    def save_all_visualizations(self) -> None:
        if self.data.empty:
            print(f"No data available for {self.currency} visualizations")
            return
            
        print(f"Generating visualizations for {self.currency}...")
        
        try:
            # Generate all plot types
            vol_surface = self.plot_volatility_surface()
            greek_surfaces = self.plot_greeks_3d()
            greeks_vs_moneyness = self.plot_greeks_vs_moneyness()
            greeks_vs_iv = self.plot_greeks_vs_iv()
            cp_ratio = self.plot_cp_ratio()
            distributions = self.plot_option_distributions()
            
            # Save all plots to HTML files
            vol_surface.write_html(self.results_dir / "volatility_surface.html")
            
            for greek, fig in greek_surfaces.items():
                fig.write_html(self.results_dir / f"greek_surface_{greek}.html")
                
            greeks_vs_moneyness.write_html(self.results_dir / "greeks_vs_moneyness.html")
            greeks_vs_iv.write_html(self.results_dir / "greeks_vs_iv.html")
            cp_ratio.write_html(self.results_dir / "cp_ratio.html")
            
            for name, fig in distributions.items():
                fig.write_html(self.results_dir / f"distribution_{name}.html")
                
            print(f"All visualizations saved to {self.results_dir}")
                
        except Exception as e:
            print(f"Error generating visualizations for {self.currency}: {e}")
            import traceback
            traceback.print_exc()

class CryptoOptionsAnalysis:
    def __init__(self, currencies: Union[List[str], str], start_date: date, end_date: date, 
                rate: float = 0.03, output_path: Optional[str] = None):
        self.currencies = [currencies.upper()] if isinstance(currencies, str) else [c.upper() for c in currencies]
        self.start_date = start_date
        self.end_date = end_date
        self.rate = rate
        self.output_path = Path(output_path) / "Option_Analysis_Results" if output_path else Path.cwd() / "Option_Analysis_Results"
        self.output_path.mkdir(exist_ok=True)
        self.results_dir = self.output_path
        self.data = {}
        self.analyzers = {}
        self.visualizers = {}
        
    def run_analysis(self):
        # Step 1: Fetch data for all currencies
        fetcher = OptionsDataFetcher(self.currencies, self.start_date, self.end_date)
        raw_data = fetcher.fetch_all_data()
        
        # Step 2: Process and analyze data for each currency
        for currency, df in raw_data.items():
            if df.empty:
                print(f"No data available for {currency}")
                continue
                
            # Create analyzer and calculate Greeks
            analyzer = OptionsAnalyzer(df, currency, self.rate, self.output_path)
            enhanced_data = analyzer.calculate_greeks()
            
            # Store processed data and analyzer
            self.data[currency] = enhanced_data
            self.analyzers[currency] = analyzer
            
            # Create visualizer
            currency_results_dir = self.results_dir / currency
            currency_results_dir.mkdir(exist_ok=True)
            visualizer = OptionsVisualizer(enhanced_data, currency, currency_results_dir)
            self.visualizers[currency] = visualizer
            
        # Step 3: Generate all outputs
        self.generate_all_outputs()
        
    def generate_all_outputs(self):
        for currency, analyzer in self.analyzers.items():
            print(f"\nGenerating outputs for {currency}...")
            
            try:
                # Get data and visualizer
                data = self.data[currency]
                visualizer = self.visualizers[currency]
                
                # Generate and save summary report
                report = analyzer.generate_summary_report()
                report_path = visualizer.results_dir / f"{currency}_analysis_report.txt"
                with open(report_path, "w") as f:
                    f.write(report)
                print(f"Summary report saved to {report_path}")
                
                # Save processed data
                data_path = visualizer.results_dir / f"{currency}_processed_data.csv"
                data.to_csv(data_path, index=False)
                print(f"Processed data saved to {data_path}")
                
                # Generate and save volume analysis
                volumes = analyzer.analyze_volumes()
                volume_path = visualizer.results_dir / f"{currency}_volume_analysis.json"
                with open(volume_path, "w") as f:
                    # Convert numpy values to Python native types for JSON serialization
                    clean_volumes = json.loads(json.dumps(volumes, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else None))
                    json.dump(clean_volumes, f, indent=4)
                print(f"Volume analysis saved to {volume_path}")
                
                # Generate and save all visualizations
                visualizer.save_all_visualizations()
                
            except Exception as e:
                print(f"Error generating outputs for {currency}: {e}")
                import traceback
                traceback.print_exc()
            
        # Create consolidated report
        self.create_consolidated_report()
        
    def create_consolidated_report(self):
        if not self.analyzers:
            print("No data available for consolidated report")
            return
            
        try:
            report = "Cryptocurrency Options Analysis Consolidated Report\n"
            report += "=" * 50 + "\n\n"
            report += f"Analysis Period: {self.start_date} to {self.end_date}\n"
            report += f"Currencies Analyzed: {', '.join(self.currencies)}\n\n"
            
            # Summary statistics table
            report += "Summary Statistics:\n"
            report += "-" * 80 + "\n"
            report += f"{'Currency':<10} {'Call Volume':<15} {'Put Volume':<15} {'P/C Ratio':<10} {'Avg Call IV':<12} {'Avg Put IV':<12}\n"
            report += "-" * 80 + "\n"
            
            for currency, analyzer in self.analyzers.items():
                volumes = analyzer.analyze_volumes()
                
                # Skip if no data
                if not volumes['calls'] or not volumes['puts']:
                    continue
                    
                call_vol = volumes['calls']['volume_usd']
                put_vol = volumes['puts']['volume_usd']
                pc_ratio = put_vol / call_vol if call_vol > 0 else 0
                call_iv = volumes['calls']['avg_iv']
                put_iv = volumes['puts']['avg_iv']
                
                report += f"{currency:<10} ${call_vol:<14,.2f} ${put_vol:<14,.2f} {pc_ratio:<10.2f} {call_iv:<12.2%} {put_iv:<12.2%}\n"
                
            report += "-" * 80 + "\n\n"
            
            # Additional comparative insights
            if len(self.analyzers) > 1:
                report += "Comparative Insights:\n"
                report += "-" * 20 + "\n"
                
                # Compare trading activity
                currencies_by_volume = sorted(self.analyzers.keys(), 
                                             key=lambda c: self.analyzers[c].analyze_volumes()['total']['volume_usd'] 
                                             if 'total' in self.analyzers[c].analyze_volumes() else 0,
                                             reverse=True)
                
                report += f"Most Active Currency: {currencies_by_volume[0]}\n"
                report += f"Ranking by Trading Volume: {', '.join(currencies_by_volume)}\n\n"
                
                # Compare volatility levels
                currencies_by_iv = sorted(self.analyzers.keys(),
                                        key=lambda c: self.data[c]['iv'].mean() if not self.data[c].empty else 0,
                                        reverse=True)
                
                report += f"Highest Average IV: {currencies_by_iv[0]}\n"
                report += f"Ranking by Average IV: {', '.join(currencies_by_iv)}\n\n"
                
            # Save consolidated report
            report_path = self.results_dir / "consolidated_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"\nConsolidated report saved to {report_path}")
            
        except Exception as e:
            print(f"Error creating consolidated report: {e}")
            import traceback
            traceback.print_exc()

def main():
    # Set date range for analysis
    end_date = date.today()
    start_date = end_date - timedelta(days=7)  # Analyze last 7 days
    
    # Specify currencies to analyze
    currencies = ['BTC', 'ETH']  # Can be extended with other supported currencies
    
    try:
        # Initialize the analysis system
        analysis = CryptoOptionsAnalysis(currencies, start_date, end_date)
        
        # Run the complete analysis pipeline
        print(f"Starting cryptocurrency options analysis for {', '.join(currencies)}...")
        print(f"Analysis period: {start_date} to {end_date}")
        
        analysis.run_analysis()
        
        print("\nAnalysis complete! All outputs have been saved to the results directory.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()     