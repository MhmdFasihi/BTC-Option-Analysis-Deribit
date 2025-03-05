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
import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor

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
    
    async def _fetch_historical_volatility_ws(self, currency: str) -> list:
        """Fetch historical volatility data from Deribit API using WebSocket"""
        try:
            # Prepare the request message
            msg = {
                "jsonrpc": "2.0",
                "id": 8387,
                "method": "public/get_historical_volatility",
                "params": {
                    "currency": currency
                }
            }
            
            # Connect to the WebSocket and send the request
            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as websocket:
                await websocket.send(json.dumps(msg))
                
                # Receive the response
                response = await websocket.recv()
                data = json.loads(response)
                
                if "result" in data:
                    return data["result"]
                else:
                    print(f"Failed to get historical volatility data for {currency}: {data.get('error', 'Unknown error')}")
                    return []
                    
        except Exception as e:
            print(f"WebSocket error fetching historical volatility for {currency}: {e}")
            return []

    def fetch_historical_volatility(self, currency: str) -> pd.DataFrame:
        """Fetch historical volatility data from Deribit API"""
        try:
            # Call the async WebSocket function
            result = asyncio.run(self._fetch_historical_volatility_ws(currency))
            
            if not result:
                print(f"No historical volatility data returned for {currency}")
                return pd.DataFrame(columns=["timestamp", "volatility", "date_time"])
            
            # Convert the data to a DataFrame
            hist_vol_data = pd.DataFrame(result, columns=["timestamp", "volatility"])
            
            # Convert timestamp to datetime
            hist_vol_data['date_time'] = pd.to_datetime(hist_vol_data['timestamp'], unit='ms')
            
            # Add date column for easier filtering
            hist_vol_data['date'] = hist_vol_data['date_time'].dt.date
            
            # Convert volatility to decimal (from percentage)
            hist_vol_data['volatility'] = hist_vol_data['volatility'] / 100
            
            # Filter data to match our analysis period
            start_timestamp = datetime_to_timestamp(dt.combine(self.start_date, dt.min.time()))
            end_timestamp = datetime_to_timestamp(dt.combine(self.end_date, dt.max.time()))
            
            hist_vol_data = hist_vol_data[
                (hist_vol_data['timestamp'] >= start_timestamp) & 
                (hist_vol_data['timestamp'] <= end_timestamp)
            ]
            
            return hist_vol_data
        except Exception as e:
            print(f"Error processing historical volatility for {currency}: {e}")
            return pd.DataFrame(columns=["timestamp", "volatility", "date_time", "date"])

    # Add a fallback method that tries both WebSocket and REST API approaches
    def fetch_historical_volatility_with_fallback(self, currency: str) -> pd.DataFrame:
        """Try multiple methods to fetch historical volatility data"""
        
        # First try WebSocket approach
        try:
            data = self.fetch_historical_volatility(currency)
            if not data.empty:
                print(f"Successfully fetched historical volatility for {currency} via WebSocket")
                return data
        except Exception as e:
            print(f"WebSocket method failed: {e}")
        
        # Fallback to REST API approach
        try:
            url = 'https://www.deribit.com/api/v2/public/get_historical_volatility'
            
            params = {
                "currency": currency
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "result" in data:
                # Convert the data to a DataFrame
                hist_vol_data = pd.DataFrame(data["result"], columns=["timestamp", "volatility"])
                
                # Convert timestamp to datetime
                hist_vol_data['date_time'] = pd.to_datetime(hist_vol_data['timestamp'], unit='ms')
                
                # Add date column
                hist_vol_data['date'] = hist_vol_data['date_time'].dt.date
                
                # Convert volatility to decimal (from percentage)
                hist_vol_data['volatility'] = hist_vol_data['volatility'] / 100
                
                # Filter data to match our analysis period
                start_timestamp = datetime_to_timestamp(dt.combine(self.start_date, dt.min.time()))
                end_timestamp = datetime_to_timestamp(dt.combine(self.end_date, dt.max.time()))
                
                hist_vol_data = hist_vol_data[
                    (hist_vol_data['timestamp'] >= start_timestamp) & 
                    (hist_vol_data['timestamp'] <= end_timestamp)
                ]
                
                print(f"Successfully fetched historical volatility for {currency} via REST API")
                return hist_vol_data
            else:
                print(f"Failed to get historical volatility data for {currency}")
                return pd.DataFrame(columns=["timestamp", "volatility", "date_time", "date"])
        except Exception as e:
            print(f"REST API method failed: {e}")
        
        # If all methods fail, return empty DataFrame and log the failure
        print(f"WARNING: Unable to fetch historical volatility data for {currency}. Both WebSocket and REST API methods failed.")
        print(f"You may need to implement a custom method for fetching historical volatility data.")
        return pd.DataFrame(columns=["timestamp", "volatility", "date_time", "date"])

    
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
        self.results_dir = Path(self.output_path) / currency
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.hist_vol_data = pd.DataFrame()  # Initialize empty historical volatility data
        self.weighted_iv = {'calls': None, 'puts': None}
        self.avg_hist_vol = None
        
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
    
    def calculate_daily_weighted_iv(self) -> pd.DataFrame:
        """Calculate weighted IV values for each day, separated by option type"""
        if self.data.empty:
            return pd.DataFrame(columns=['date', 'weighted_iv_calls', 'weighted_iv_puts', 'weighted_iv_all'])
        
        # Ensure we have date column
        self.data['date'] = pd.to_datetime(self.data['date_time']).dt.date
        
        daily_iv = []
        
        # Get unique dates
        unique_dates = sorted(self.data['date'].unique())
        
        for date in unique_dates:
            day_data = self.data[self.data['date'] == date]
            
            # Calls
            calls = day_data[day_data['option_type'] == 'call']
            weighted_iv_calls = None
            if not calls.empty and calls['volume_usd'].sum() > 0:
                weighted_iv_calls = (calls['iv'] * calls['volume_usd']).sum() / calls['volume_usd'].sum()
            
            # Puts
            puts = day_data[day_data['option_type'] == 'put']
            weighted_iv_puts = None
            if not puts.empty and puts['volume_usd'].sum() > 0:
                weighted_iv_puts = (puts['iv'] * puts['volume_usd']).sum() / puts['volume_usd'].sum()
            
            # All options
            weighted_iv_all = None
            if day_data['volume_usd'].sum() > 0:
                weighted_iv_all = (day_data['iv'] * day_data['volume_usd']).sum() / day_data['volume_usd'].sum()
            
            daily_iv.append({
                'date': date,
                'weighted_iv_calls': weighted_iv_calls * 100 if weighted_iv_calls is not None else None,  # Convert to percentage
                'weighted_iv_puts': weighted_iv_puts * 100 if weighted_iv_puts is not None else None,
                'weighted_iv_all': weighted_iv_all * 100 if weighted_iv_all is not None else None
            })
        
        return pd.DataFrame(daily_iv)

        
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
    
    def calculate_weighted_iv(self) -> dict:
        """Calculate volume-weighted implied volatility for calls and puts"""
        if self.data.empty:
            return {'calls': None, 'puts': None}
        
        calls = self.data[self.data['option_type'] == 'call']
        puts = self.data[self.data['option_type'] == 'put']
        
        weighted_iv = {}
        
        # Calculate weighted IV for calls
        if not calls.empty and calls['volume_usd'].sum() > 0:
            weighted_iv['calls'] = (calls['iv'] * calls['volume_usd']).sum() / calls['volume_usd'].sum()
        else:
            weighted_iv['calls'] = None
        
        # Calculate weighted IV for puts
        if not puts.empty and puts['volume_usd'].sum() > 0:
            weighted_iv['puts'] = (puts['iv'] * puts['volume_usd']).sum() / puts['volume_usd'].sum()
        else:
            weighted_iv['puts'] = None
        
        return weighted_iv
        
    def fetch_and_process_historical_volatility(self, fetcher: OptionsDataFetcher) -> float:
        """Fetch historical volatility and calculate average"""
        self.hist_vol_data = fetcher.fetch_historical_volatility_with_fallback(self.currency)
        
        if not self.hist_vol_data.empty:
            return self.hist_vol_data['volatility'].mean()
        else:
            print(f"No historical volatility data available for {self.currency}")
            return None

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
    
    def plot_iv_skew_by_maturity(self) -> go.Figure:
        """
        Create an enhanced plot showing IV skew by moneyness for different maturities,
        with improved readability and analytical features.
        """
        from plotly.subplots import make_subplots
        import numpy as np
        from scipy.stats import norm
        from scipy import signal

        if self.data.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No data available for {self.currency} IV Skew")
            return fig
        
        # Create subplot with two rows (calls and puts)
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            subplot_titles=(f"{self.currency} Call Options IV Skew", 
                                        f"{self.currency} Put Options IV Skew"),
                            vertical_spacing=0.08,
                            row_heights=[0.5, 0.5])
        
        # Set reasonable IV cap to avoid extreme outliers distorting the visualization
        max_iv_display = 250  # Cap at 250%
        
        # Get current price for reference
        current_price = self.data['index_price'].iloc[-1]
        
        # Define a smoothing function for IV curves
        def smooth_curve(x, y, window=5):
            if len(x) < window:
                return x, y
            try:
                # Use Savitzky-Golay filter for smoothing
                y_smooth = signal.savgol_filter(y, 
                                            window_length=min(window, len(y) - (1 if len(y) % 2 == 0 else 0)), 
                                            polyorder=min(3, window-1))
                return x, y_smooth
            except Exception as e:
                print(f"Smoothing error: {e}")
                return x, y
        
        # Helper to get median IV for a particular option type and moneyness range
        def get_atm_iv(option_type, moneyness_min=0.95, moneyness_max=1.05):
            subset = self.data[(self.data['option_type'] == option_type) & 
                            (self.data['moneyness'] >= moneyness_min) & 
                            (self.data['moneyness'] <= moneyness_max)]
            if subset.empty:
                return None
            return np.median(subset['iv']) * 100
        
        call_atm_iv = get_atm_iv('call')
        put_atm_iv = get_atm_iv('put')
        
        # Calculate 25-delta put/call skew if we have the data
        skew_metrics = {}
        if 'delta' in self.data.columns:
            for opt_type in ['call', 'put']:
                # Filter near-term options (< 30 days)
                near_term = self.data[self.data['option_type'] == opt_type]
                if not near_term.empty and 'delta' in near_term.columns:
                    near_term['abs_delta'] = near_term['delta'].abs()
                    # Get options closest to 25 delta
                    delta_25 = near_term.iloc[(near_term['abs_delta'] - 0.25).abs().argsort()[:10]]
                    if not delta_25.empty:
                        skew_metrics[f"{opt_type}_25delta_iv"] = delta_25['iv'].mean() * 100

        # Calculate skew as difference between 25-delta put and 25-delta call IV
        if 'put_25delta_iv' in skew_metrics and 'call_25delta_iv' in skew_metrics:
            skew_metrics['skew_25delta'] = skew_metrics['put_25delta_iv'] - skew_metrics['call_25delta_iv']
        
        # Get historical vol from analyzer if available
        hist_vol = None
        if hasattr(self, 'analyzer') and hasattr(self.analyzer, 'avg_hist_vol') and self.analyzer.avg_hist_vol is not None:
            hist_vol = self.analyzer.avg_hist_vol * 100  # Convert to percentage
        
        # Process data for each option type
        for option_idx, (option_type, color, row) in enumerate([('call', 'blue', 1), ('put', 'red', 2)]):
            type_data = self.data[self.data['option_type'] == option_type]
            
            if type_data.empty:
                continue
            
            # Get the most active maturity dates (top 5 by number of contracts)
            maturity_counts = type_data['maturity_date'].value_counts().nlargest(5)
            top_maturities = maturity_counts.index.tolist()
            
            # Plot IV curve for each maturity
            for maturity in top_maturities:
                # Filter data for this maturity
                maturity_data = type_data[type_data['maturity_date'] == maturity]
                
                # Calculate days to maturity for labeling
                # Convert both to date objects to fix the type mismatch
                current_date = pd.Timestamp.now().date()
                maturity_date = maturity.date() if hasattr(maturity, 'date') else maturity
                
                days_to_maturity = (maturity_date - current_date).days
                if days_to_maturity < 0:
                    continue  # Skip expired options
                
                # Filter out extreme IV values
                maturity_data = maturity_data[maturity_data['iv'] * 100 <= max_iv_display]
                
                # Need at least 3 points for a meaningful curve
                if len(maturity_data) < 3:
                    continue
                    
                # Sort by moneyness for smooth curve
                maturity_data = maturity_data.sort_values('moneyness')
                
                # Smooth the curve
                x_values = maturity_data['moneyness'].values
                y_values = maturity_data['iv'].values * 100  # Convert to percentage
                
                # Apply smoothing if we have enough data points
                if len(x_values) >= 5:
                    x_smooth, y_smooth = smooth_curve(x_values, y_values, window=min(11, len(x_values) - (0 if len(x_values) % 2 == 1 else 1)))
                else:
                    x_smooth, y_smooth = x_values, y_values
                
                # Determine line style based on nearness to expiry
                if days_to_maturity <= 7:
                    line_style = 'solid'
                    width = 3
                elif days_to_maturity <= 30:
                    line_style = 'dash'
                    width = 2
                else:
                    line_style = 'dot'
                    width = 1.5
                
                # Add the IV curve to appropriate subplot
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name=f"{option_type.capitalize()} {maturity_date.strftime('%Y-%m-%d')} ({days_to_maturity}d)",
                    line=dict(
                        color=color,
                        width=width,
                        dash=line_style
                    ),
                    opacity=0.7
                ), row=row, col=1)
        
        # Add reference lines to both subplots
        for row in [1, 2]:
            # Add vertical line at moneyness = 1.0 (at-the-money)
            fig.add_vline(
                x=1.0,
                line_dash="dash",
                line_color="green",
                row=row, col=1
            )
            
            # Add horizontal line for historical volatility if available
            if hist_vol is not None:
                fig.add_hline(
                    y=hist_vol,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"30d Historical Vol: {hist_vol:.1f}%",
                    annotation_position="top right",
                    row=row, col=1
                )
            
            # Add horizontal line for ATM IV
            atm_iv = call_atm_iv if row == 1 else put_atm_iv
            if atm_iv is not None:
                fig.add_hline(
                    y=atm_iv,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text=f"ATM IV: {atm_iv:.1f}%",
                    annotation_position="top left",
                    row=row, col=1
                )
        
        # Add skew metric annotation if available
        if 'skew_25delta' in skew_metrics:
            fig.add_annotation(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"25-Delta Put-Call Skew: {skew_metrics['skew_25delta']:.1f}%",
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        # Update layout for better readability
        fig.update_layout(
            title=f"{self.currency} IV Skew by Maturity and Option Type",
            xaxis_title="",  # Will be set for the shared x axis
            xaxis2_title="Moneyness (Spot/Strike)",
            showlegend=True,
            legend=dict(
                title="Option Type - Maturity (Days)",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.15
            ),
            hovermode="closest",
            plot_bgcolor='white',
            width=1200,
            height=900,  # Increased height for the dual subplot layout
            margin=dict(r=150)  # Add right margin for legend
        )
        
        # Set consistent y-axis ranges for both subplots
        y_max = min(max_iv_display, self.data['iv'].max() * 100 * 1.1)  # 10% padding but capped
        y_min = max(0, self.data['iv'].min() * 100 * 0.9)  # 10% padding but minimum 0
        
        fig.update_yaxes(title_text="Implied Volatility (%)", range=[y_min, y_max], row=1, col=1)
        fig.update_yaxes(title_text="Implied Volatility (%)", range=[y_min, y_max], row=2, col=1)
        
        # Set consistent x-axis range
        x_min = max(0.4, self.data['moneyness'].min() * 0.9)
        x_max = min(2.5, self.data['moneyness'].max() * 1.1)
        
        fig.update_xaxes(range=[x_min, x_max], row=1, col=1)
        fig.update_xaxes(range=[x_min, x_max], row=2, col=1)
        
        # Add annotations explaining IV skew to each panel
        # For call panel
        fig.add_annotation(
            x=0.85,
            y=y_max * 0.9,
            text="⬅️ ITM Calls",
            showarrow=False,
            font=dict(size=10),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1,
            row=1, col=1
        )
        
        fig.add_annotation(
            x=1.15,
            y=y_max * 0.9,
            text="OTM Calls ➡️",
            showarrow=False,
            font=dict(size=10),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1,
            row=1, col=1
        )
        
        # For put panel
        fig.add_annotation(
            x=0.85,
            y=y_max * 0.9,
            text="⬅️ OTM Puts<br>(Higher IV typically)",
            showarrow=False,
            font=dict(size=10),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            row=2, col=1
        )
        
        fig.add_annotation(
            x=1.15,
            y=y_max * 0.9,
            text="ITM Puts ➡️",
            showarrow=False,
            font=dict(size=10),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            row=2, col=1
        )
        
        return fig

    
    def plot_weighted_iv_timeseries(self) -> go.Figure:
        """Create a time series plot of daily weighted IV values"""
        if not hasattr(self, 'analyzer') or self.analyzer is None:
            fig = go.Figure()
            fig.update_layout(title=f"No analyzer available for {self.currency} Weighted IV")
            return fig
        
        daily_iv = self.analyzer.calculate_daily_weighted_iv()
        
        if daily_iv.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No data available for {self.currency} Weighted IV")
            return fig
        
        fig = go.Figure()
        
        # Add line for calls (blue)
        if not daily_iv['weighted_iv_calls'].isna().all():
            fig.add_trace(go.Scatter(
                x=daily_iv['date'],
                y=daily_iv['weighted_iv_calls'],
                mode='lines+markers',
                name='Weighted IV (Calls)',
                line=dict(color='blue', width=2),
                marker=dict(size=8, color='blue')
            ))
        
        # Add line for puts (red)
        if not daily_iv['weighted_iv_puts'].isna().all():
            fig.add_trace(go.Scatter(
                x=daily_iv['date'],
                y=daily_iv['weighted_iv_puts'],
                mode='lines+markers',
                name='Weighted IV (Puts)',
                line=dict(color='red', width=2),
                marker=dict(size=8, color='red')
            ))
        
        # Add line for all options (purple)
        if not daily_iv['weighted_iv_all'].isna().all():
            fig.add_trace(go.Scatter(
                x=daily_iv['date'],
                y=daily_iv['weighted_iv_all'],
                mode='lines+markers',
                name='Weighted IV (All)',
                line=dict(color='purple', width=2),
                marker=dict(size=8, color='purple')
            ))
        
        # Add historical volatility reference line if available
        if hasattr(self.analyzer, 'hist_vol_data') and not self.analyzer.hist_vol_data.empty:
            # Process historical volatility data for plotting
            hist_vol = self.analyzer.hist_vol_data.copy()
            hist_vol['date'] = pd.to_datetime(hist_vol['date_time']).dt.date
            
            # Calculate daily average
            daily_hist_vol = hist_vol.groupby('date')['volatility'].mean().reset_index()
            
            # Convert to percentage for consistency with IV
            daily_hist_vol['volatility'] = daily_hist_vol['volatility'] * 100
            
            fig.add_trace(go.Scatter(
                x=daily_hist_vol['date'],
                y=daily_hist_vol['volatility'],
                mode='lines+markers',
                name='Historical Volatility',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=8, color='green')
            ))
        
        fig.update_layout(
            title=f"{self.currency} Weighted Implied Volatility Over Time",
            xaxis_title="Date",
            yaxis_title="Weighted IV (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1
            ),
            xaxis=dict(
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            width=1200,
            height=600
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
    
    def plot_greek_iv_ttm_surfaces(self) -> Dict[str, go.Figure]:
        """Create 3D surface plots for Greeks with IV and Time to Maturity as axes"""
        if self.data.empty or 'delta' not in self.data.columns:
            return {}
            
        figures = {}
        
        for greek in ['delta', 'gamma', 'vega', 'theta']:
            # Separate plots for calls and puts
            for option_type, title_prefix, color in [
                ('call', 'Call', 'blue'), 
                ('put', 'Put', 'red')
            ]:
                # Filter data for this option type
                filtered_data = self.data[self.data['option_type'] == option_type]
                
                if filtered_data.empty:
                    continue
                    
                fig = go.Figure()
                
                fig.add_trace(go.Scatter3d(
                    x=filtered_data['iv'],
                    y=filtered_data['time_to_maturity'],
                    z=filtered_data[greek],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=filtered_data[greek],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    name=f'{greek.capitalize()} vs IV & TTM'
                ))
                
                fig.update_layout(
                    title=f"{self.currency} {title_prefix} {greek.capitalize()} Surface (IV & TTM)",
                    scene=dict(
                        xaxis_title="Implied Volatility",
                        yaxis_title="Time to Maturity (days)",
                        zaxis_title=greek.capitalize()
                    ),
                    width=1000,
                    height=800
                )
                
                figures[f"{option_type}_{greek}_iv_ttm"] = fig
        
        return figures

        
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
        
        # 1. Enhanced Strike Price Distribution with separated dots
        if not calls.empty or not puts.empty:
            fig_strike = go.Figure()
            
            # Get weighted IV and historical volatility from analyzer
            weighted_iv = getattr(getattr(self, 'analyzer', None), 'weighted_iv', {'calls': None, 'puts': None})
            avg_hist_vol = getattr(getattr(self, 'analyzer', None), 'avg_hist_vol', None)
            
            # Add violin plots with statistical information
            for opt_type, data, stats, color, y_offset in [
                ('Calls', calls, call_stats, 'blue', -0.7),  # Position calls at bottom
                ('Puts', puts, put_stats, 'red', 0.7)        # Position puts at top
            ]:
                if data.empty:
                    continue
                    
                # Add violin plot
                fig_strike.add_trace(go.Violin(
                    x=data['strike_price'],
                    y=[y_offset] * len(data),  # Offset y-position
                    name=opt_type,
                    side='positive' if opt_type == 'Calls' else 'negative',
                    line_color=color,
                    fillcolor=f'rgba{(0, 0, 255, 0.2) if color=="blue" else (255, 0, 0, 0.2)}',
                    opacity=0.6,
                    points=False,
                    box=dict(visible=True),
                    meanline=dict(visible=True),
                    hoverinfo='skip',
                    orientation='h'  # Horizontal orientation
                ))
                
                # Add scatter points for trades (with y-offset)
                fig_strike.add_trace(go.Scatter(
                    x=data['strike_price'],
                    y=[y_offset] * len(data),  # Offset y-position
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
                
                # Add statistical annotations to the right of the chart, not overlapping
                if stats:
                    # Determine which weighted IV to use
                    w_iv = weighted_iv.get('calls' if opt_type == 'Calls' else 'puts')
                    w_iv_str = f"{w_iv:.1%}" if w_iv is not None else "N/A"
                    
                    stats_text = (
                        f"<b>{opt_type} Statistics</b><br>" +
                        f"Mean Strike: ${stats['mean']:,.0f}<br>" +
                        f"Median Strike: ${stats['median']:,.0f}<br>" +
                        f"Std Dev: ${stats['std']:,.0f}<br>" +
                        f"Volume Weighted Avg: ${stats['volume_weighted_avg']:,.0f}<br>" +
                        f"Total Volume: ${stats['total_volume']:,.2f}<br>" +
                        f"Contracts: {stats['contract_count']:,}<br>" +
                        f"Avg IV: {stats['iv_mean']:.1%}<br>" +
                        f"Weighted IV: {w_iv_str}"
                    )
                    
                    if avg_hist_vol is not None:
                        stats_text += f"<br>Avg Hist Vol: {avg_hist_vol:.1%}"
                    
                    if stats['delta_mean'] is not None:
                        stats_text += f"<br>Avg Delta: {stats['delta_mean']:.3f}"
                    
                    # Position the statistics box outside the plot area on the right
                    fig_strike.add_annotation(
                        x=1.15,  # Position to the right of the plot
                        y=y_offset,  # Keep y-offset consistent with the plot
                        xref="paper",  # Use paper coordinates for x
                        yref="y",      # Use data coordinates for y
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
            
            # Update layout to accommodate statistics boxes on the right
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
                    range=[current_price * 0.5, current_price * 2],
                    domain=[0, 0.85]  # Reduce x-axis width to make room for stat boxes
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
                    zerolinewidth=1,
                    ticktext=["Calls", "", "Puts"],  # Label the y-axis
                    tickvals=[-0.7, 0, 0.7],
                    range=[-1.5, 1.5]  # Expand y-axis range
                ),
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=0.95,
                    bordercolor="black",
                    borderwidth=1,
                    bgcolor="white"
                ),
                hovermode='closest',
                plot_bgcolor='white',
                width=1400,
                height=800,
                margin=dict(t=100, b=50, l=50, r=300)  # Extend right margin for stat boxes
            )
            figures['strike'] = fig_strike

        # 3. Separate Call and Put Strike Box Plots (Modified from distribution_strike_box)
        if not calls.empty:
            # Create Call Strike Box Plot
            fig_call_box = go.Figure()
            
            # Group calls by maturity date
            maturity_dates = sorted(calls['maturity_date'].unique())
            
            for date in maturity_dates:
                date_data = calls[calls['maturity_date'] == date]['strike_price']
                fig_call_box.add_trace(go.Box(
                    y=date_data,
                    name=date.strftime('%Y-%m-%d'),
                    boxpoints='outliers',
                    jitter=0.3,
                    marker=dict(color='blue', opacity=0.7)
                ))
            
            # Add current price line
            fig_call_box.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Current Price: ${current_price:,.0f}"
            )
            
            fig_call_box.update_layout(
                title=dict(text=f"{self.currency} Call Strike Price Distribution by Maturity Date", x=0.5),
                xaxis_title="Maturity Date",
                yaxis_title="Strike Price",
                xaxis_tickangle=45,
                height=800,
                yaxis=dict(tickformat="$,.0f"),
                plot_bgcolor='white'
            )
            
            figures['call_strike_box'] = fig_call_box
        
        if not puts.empty:
            # Create Put Strike Box Plot
            fig_put_box = go.Figure()
            
            # Group puts by maturity date
            maturity_dates = sorted(puts['maturity_date'].unique())
            
            for date in maturity_dates:
                date_data = puts[puts['maturity_date'] == date]['strike_price']
                fig_put_box.add_trace(go.Box(
                    y=date_data,
                    name=date.strftime('%Y-%m-%d'),
                    boxpoints='outliers',
                    jitter=0.3,
                    marker=dict(color='red', opacity=0.7)
                ))
            
            # Add current price line
            fig_put_box.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Current Price: ${current_price:,.0f}"
            )
            
            fig_put_box.update_layout(
                title=dict(text=f"{self.currency} Put Strike Price Distribution by Maturity Date", x=0.5),
                xaxis_title="Maturity Date",
                yaxis_title="Strike Price",
                xaxis_tickangle=45,
                height=800,
                yaxis=dict(tickformat="$,.0f"),
                plot_bgcolor='white'
            )
            
            figures['put_strike_box'] = fig_put_box
            
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
            greek_iv_ttm_surfaces = self.plot_greek_iv_ttm_surfaces()
            greeks_vs_moneyness = self.plot_greeks_vs_moneyness()
            greeks_vs_iv = self.plot_greeks_vs_iv()
            cp_ratio = self.plot_cp_ratio()
            distributions = self.plot_option_distributions()
            weighted_iv_timeseries = self.plot_weighted_iv_timeseries()
            iv_skew_by_maturity = self.plot_iv_skew_by_maturity()  # Add the new plot
            
            # Create results directory if there are visualizations
            self.results_dir.mkdir(exist_ok=True, parents=True)
            
            # Save all plots to HTML files
            vol_surface.write_html(self.results_dir / "volatility_surface.html")
            
            for greek, fig in greek_surfaces.items():
                fig.write_html(self.results_dir / f"greek_surface_{greek}.html")
                
            for name, fig in greek_iv_ttm_surfaces.items():
                fig.write_html(self.results_dir / f"{name}.html")
                
            greeks_vs_moneyness.write_html(self.results_dir / "greeks_vs_moneyness.html")
            greeks_vs_iv.write_html(self.results_dir / "greeks_vs_iv.html")
            cp_ratio.write_html(self.results_dir / "cp_ratio.html")
            weighted_iv_timeseries.write_html(self.results_dir / "weighted_iv_timeseries.html")
            iv_skew_by_maturity.write_html(self.results_dir / "iv_skew_by_maturity.html")  # Save the new plot
            
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
            
            # Calculate weighted IV
            analyzer.weighted_iv = analyzer.calculate_weighted_iv()
            
            # Fetch and calculate average historical volatility
            analyzer.avg_hist_vol = analyzer.fetch_and_process_historical_volatility(fetcher)
            
            # Store processed data and analyzer
            self.data[currency] = enhanced_data
            self.analyzers[currency] = analyzer
            
            # Create visualizer with updated directory structure
            currency_results_dir = self.results_dir / currency
            currency_results_dir.mkdir(exist_ok=True)
            visualizer = OptionsVisualizer(enhanced_data, currency, currency_results_dir)
            visualizer.analyzer = analyzer  # Pass analyzer reference to visualizer
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
                
                # Skip if data is empty
                if data.empty:
                    print(f"No data available for {currency}, skipping output generation")
                    continue
                
                # Create currency directory only if we have data
                currency_dir = self.results_dir / currency
                
                # Generate and save summary report
                report = analyzer.generate_summary_report()
                
                # Check if there's meaningful content in the report
                if "No data available" not in report:
                    currency_dir.mkdir(exist_ok=True)
                    report_path = currency_dir / f"{currency}_analysis_report.txt"
                    with open(report_path, "w") as f:
                        f.write(report)
                    print(f"Summary report saved to {report_path}")
                    
                    # Save processed data
                    data_path = currency_dir / f"{currency}_processed_data.csv"
                    data.to_csv(data_path, index=False)
                    print(f"Processed data saved to {data_path}")
                    
                    # Generate and save volume analysis
                    volumes = analyzer.analyze_volumes()
                    volume_path = currency_dir / f"{currency}_volume_analysis.json"
                    with open(volume_path, "w") as f:
                        # Convert numpy values to Python native types for JSON serialization
                        clean_volumes = json.loads(json.dumps(volumes, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else None))
                        json.dump(clean_volumes, f, indent=4)
                    print(f"Volume analysis saved to {volume_path}")
                    
                    # Generate and save all visualizations
                    visualizer.save_all_visualizations()
                    
                    # Clean up - remove currency directory if it's empty
                    if not any(currency_dir.iterdir()):
                        currency_dir.rmdir()
                        print(f"Removed empty directory: {currency_dir}")
                
            except Exception as e:
                print(f"Error generating outputs for {currency}: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to clean up if exception occurred
                currency_dir = self.results_dir / currency
                if currency_dir.exists() and not any(currency_dir.iterdir()):
                    try:
                        currency_dir.rmdir()
                        print(f"Removed empty directory: {currency_dir}")
                    except:
                        pass
        
        # Create consolidated report
        self.create_consolidated_report()
        
        # Clean up underlying_assets folder if it exists and is empty
        underlying_assets_dir = self.results_dir / "underlying_assets"
        if underlying_assets_dir.exists() and not any(underlying_assets_dir.iterdir()):
            try:
                underlying_assets_dir.rmdir()
                print(f"Removed empty directory: {underlying_assets_dir}")
            except:
                pass
        
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
    start_date = end_date - timedelta(days=365)  # Analyze last 7 days
    
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
