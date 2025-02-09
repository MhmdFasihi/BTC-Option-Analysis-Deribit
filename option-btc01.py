import numpy as np
import pandas as pd
import requests
from datetime import datetime as dt
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from typing import Optional, Dict, List, Tuple
import warnings
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os

warnings.filterwarnings('ignore')

def datetime_to_timestamp(datetime_obj: dt) -> int:
    """Convert datetime to millisecond timestamp"""
    return int(dt.timestamp(datetime_obj) * 1000)

class OptionsAnalyzer:
    """Enhanced options data analysis class with fixed Greeks calculations"""

    def __init__(self, currency: str, start_date: date, end_date: date, rate: float = 0.03, output_path: Optional[str] = None):
        self.currency = currency.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.rate = rate
        self.data = None
        self.output_path = output_path or os.getcwd()  # Default to current working directory
        self.results_dir = Path(self.output_path) / "results"
        self.results_dir.mkdir(exist_ok=True)  # Create results directory if it doesn't exist
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.currency not in ['BTC', 'ETH']:
            raise ValueError("Currency must be BTC or ETH")
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
        if (self.end_date - self.start_date).days > 30:
            raise ValueError("Date range cannot exceed 30 days")

    def _fetch_trades(self, start_timestamp: int, end_timestamp: int) -> List[Dict]:
        url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'
        params = {
            "currency": self.currency,
            "kind": "option",
            "count": 10000,
            "include_old": True,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get("result", {}).get("trades", [])
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

    def _parse_instrument_name(self, name: str) -> Tuple[str, str, float, str]:
        """Parse instrument name to extract details"""
        try:
            parts = name.split('-')
            currency = parts[0]
            maturity = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            return maturity, strike, option_type
        except Exception as e:
            print(f"Error parsing instrument name {name}: {e}")
            return None, None, None

    def _calculate_greeks(self, row: pd.Series) -> Tuple[float, float, float, float]:
        """Vectorized Greeks calculation with fixed put/call handling"""
        S = row['index_price']
        K = row['strike_price']
        T = max(row['time_to_maturity'] / 365, 1e-6)
        sigma = row['iv']
        r = self.rate
        is_call = row['option_type'] == 'call'

        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Fixed delta calculation
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)  # Fixed put delta calculation

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01

        # Theta calculation
        theta_common = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))
        if is_call:
            theta = theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)

        return delta, gamma, vega, theta

    def fetch_data(self) -> pd.DataFrame:
        """Fetch and process options data with fixed string operations"""
        print("Fetching data...")
        start_ts = datetime_to_timestamp(dt.combine(self.start_date, dt.min.time()))
        end_ts = datetime_to_timestamp(dt.combine(self.end_date, dt.max.time()))

        trades = self._fetch_trades(start_ts, end_ts)
        if not trades:
            raise ValueError("No data retrieved")

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        print("Processing data...")
        # Convert timestamp to datetime
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Process instrument names
        instrument_details = []
        for name in df['instrument_name']:
            maturity, strike, option_type = self._parse_instrument_name(name)
            instrument_details.append((maturity, strike, option_type))

        # Add parsed details to DataFrame
        df['maturity_date'] = [details[0] for details in instrument_details]
        df['strike_price'] = [details[1] for details in instrument_details]
        df['option_type'] = [details[2] for details in instrument_details]

        # Convert maturity date to datetime
        df['maturity_date'] = pd.to_datetime(df['maturity_date'], format='%d%b%y')

        # Calculate key metrics
        df['time_to_maturity'] = (df['maturity_date'] - df['date_time']).dt.total_seconds() / 86400
        df['moneyness'] = df['index_price'] / df['strike_price']
        df['iv'] = df['iv'] / 100

        # Calculate Greeks
        print("Calculating Greeks...")
        greeks = df.apply(self._calculate_greeks, axis=1, result_type='expand')
        greeks.columns = ['delta', 'gamma', 'vega', 'theta']
        df = pd.concat([df, greeks], axis=1)

        # Add option type indicator and calculate daily call/put ratio
        df['is_call'] = df['option_type'] == 'call'
        df['date'] = df['date_time'].dt.date

        # Calculate daily call/put volume ratio
        daily_cp_ratio = df.groupby('date').agg({
            'is_call': ['sum', 'size']
        }).reset_index()
        daily_cp_ratio.columns = ['date', 'calls', 'total']
        daily_cp_ratio['puts'] = daily_cp_ratio['total'] - daily_cp_ratio['calls']
        daily_cp_ratio['cp_ratio'] = daily_cp_ratio['calls'] / daily_cp_ratio['puts']

        # Merge call/put ratio back to main dataframe
        df = df.merge(daily_cp_ratio[['date', 'cp_ratio']], on='date', how='left')

        # Select final columns
        columns = ['date_time', 'instrument_name', 'price', 'index_price', 'strike_price',
                  'option_type', 'iv', 'moneyness', 'time_to_maturity', 'delta', 'gamma',
                  'vega', 'theta', 'is_call', 'cp_ratio']

        self.data = df[columns].copy()

        # Print summary statistics
        print("\nData Summary:")
        print(f"Total options: {len(df)}")
        print(f"Calls: {df['is_call'].sum()}")
        print(f"Puts: {(~df['is_call']).sum()}")
        print("\nCall/Put Ratio Summary:")
        print(daily_cp_ratio[['date', 'cp_ratio']].to_string(index=False))
        print("\nDelta ranges:")
        print(f"Call deltas: {df[df['is_call']]['delta'].describe()}")
        print(f"Put deltas: {df[~df['is_call']]['delta'].describe()}")

        return self.data

    def plot_cp_ratio(self) -> go.Figure:
        """Generate call/put ratio visualization"""
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")

        print("Generating call/put ratio plot...")
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

    def plot_volatility_surface(self) -> go.Figure:
        """Generate volatility surface plot"""
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")

        print("Generating volatility surface plot...")
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
                xaxis_title="Moneyness",
                yaxis_title="Time to Maturity (days)",
                zaxis_title="Implied Volatility"
            ),
            width=1000,
            height=800
        )

        return fig

    def plot_greeks_3d(self) -> List[go.Figure]:
        """Generate 3D plots for each Greek vs time to maturity"""
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")

        print("Generating 3D Greek plots...")
        figures = []

        for greek in ['delta', 'gamma', 'vega', 'theta']:
            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=self.data['time_to_maturity'],
                y=self.data['iv'],
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
                    yaxis_title="Implied Volatility",
                    zaxis_title=greek.capitalize()
                ),
                width=1000,
                height=800
            )

            figures.append(fig)

        return figures

    def plot_greeks_vs_moneyness(self) -> go.Figure:
        """Generate Greeks vs moneyness visualization"""
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")

        print("Generating Greeks vs moneyness plot...")
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
            xaxis_title="Moneyness",
            yaxis_title="Greek Value",
            width=1000,
            height=600
        )

        return fig

    def plot_greeks_vs_iv(self) -> go.Figure:
        """Generate Greeks vs IV visualization"""
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")

        print("Generating Greeks vs IV plot...")
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

class OptionDataAnalyzer:
    def __init__(self, days_back: int = 7, cache_dir: str = "cache", output_path: Optional[str] = None):
        """
        Initialize with dynamic date range based on current date.

        Args:
            days_back (int): Number of days to look back from today
            cache_dir (str): Directory for caching data
            output_path (str): Directory to save results
        """
        self.end_date = date.today()
        self.start_date = self.end_date - timedelta(days=days_back)
        self.currency = "BTC"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.output_path = output_path or os.getcwd()  # Default to current working directory
        self.results_dir = Path(self.output_path) / "results"
        self.results_dir.mkdir(exist_ok=True)  # Create results directory if it doesn't exist

        logging.basicConfig(
            filename=f"option_data_{self.currency}_{self.start_date}_{self.end_date}.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.session = requests.Session()
        self.cache_file = self.cache_dir / f"{self.currency}_options_{self.start_date}_{self.end_date}.csv"

    @staticmethod
    def _datetime_to_timestamp(datetime_obj: dt) -> int:
        if isinstance(datetime_obj, date):
            datetime_obj = dt.combine(datetime_obj, dt.min.time())
        return int(dt.timestamp(datetime_obj) * 1000)

    def _get_data_from_api(self, start_ts: int, end_ts: int) -> list:
        params = {
            "currency": self.currency,
            "kind": "option",
            "count": 1000,
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
                    logging.error(f"Failed to get data: {e}")
                    return []
                time.sleep(0.5 * (2 ** attempt))
        return []

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and calculate all necessary fields for the data."""
        if data.empty:
            return data

        # Ensure date columns are datetime
        data['maturity_date'] = pd.to_datetime(data['maturity_date'])
        data['date_time'] = pd.to_datetime(data['date_time'])

        # Calculate time to maturity in days
        data['time_to_maturity'] = (data['maturity_date'] - data['date_time']).dt.total_seconds() / (24 * 60 * 60)
        data['time_to_maturity'] = data['time_to_maturity'].round(2)

        # Create IV bins (10% intervals)
        data['iv_bin'] = pd.cut(data['iv'], bins=np.arange(0, 1.1, 0.1), right=False)

        return data

    def _process_chunk(self, chunk: list) -> pd.DataFrame:
        if not chunk:
            return pd.DataFrame()

        df = pd.DataFrame(chunk)

        df[['kind', 'maturity', 'strike', 'option_type']] = (
            df['instrument_name'].str.split('-', expand=True)[[0, 1, 2, 3]]
        )

        df['maturity_date'] = pd.to_datetime(df['maturity'], format='%d%b%y')
        df['strike_price'] = df['strike'].astype(float)
        df['option_type'] = df['option_type'].str.lower()
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['moneyness'] = df['index_price'] / df['strike_price']
        df['price'] = (df['price'] * df['index_price']).round(2)
        df['iv'] = df['iv'] / 100

        return df[df['option_type'] == 'c']

    def get_option_data(self) -> pd.DataFrame:
        if self.cache_file.exists():
            logging.info("Loading data from cache...")
            data = pd.read_csv(self.cache_file)
            return self._process_data(data)

        logging.info("Fetching new data from API...")

        date_chunks = []
        current_date = self.start_date
        while current_date < self.end_date:
            next_date = min(current_date + timedelta(days=1), self.end_date)
            date_chunks.append((
                self._datetime_to_timestamp(current_date),
                self._datetime_to_timestamp(next_date)
            ))
            current_date = next_date

        all_trades = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self._get_data_from_api, start_ts, end_ts)
                for start_ts, end_ts in date_chunks
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching data"):
                trades = future.result()
                if trades:
                    all_trades.extend(trades)

        if not all_trades:
            logging.warning("No data retrieved")
            return pd.DataFrame()

        df = self._process_chunk(all_trades)
        df.to_csv(self.cache_file, index=False)
        return self._process_data(df)

    def create_visualizations(self, data: pd.DataFrame):
        if data.empty:
            logging.warning("No data available for visualization")
            return

        # 1. Option Price Over Time
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=data.sort_values('date_time')['date_time'],
            y=data.sort_values('date_time')['price'],
            mode='lines',
            name='Option Price'
        ))
        fig1.update_layout(
            title='Option Price Over Time',
            xaxis_title='Date',
            yaxis_title='Option Price',
            height=800
        )
        fig1.write_html(self.results_dir / "option_price_time.html")

        # 2. Strike Price Distribution by Maturity Date (Box Plot)
        fig2 = go.Figure()
        dates = sorted(data['maturity_date'].unique())

        for date in dates:
            date_data = data[data['maturity_date'] == date]['strike_price']
            fig2.add_trace(go.Box(
                y=date_data,
                name=date.strftime('%Y-%m-%d'),
                boxpoints='outliers'
            ))

        fig2.update_layout(
            title='Strike Price Distribution by Maturity Date',
            xaxis_title='Maturity Date',
            yaxis_title='Strike Price',
            xaxis_tickangle=45,
            height=800
        )
        fig2.write_html(self.results_dir / "strike_distribution.html")

        # 3. IV Distribution with 10% Bins
        iv_bins = pd.cut(data['iv'], bins=np.arange(0, 1.1, 0.1))
        iv_counts = iv_bins.value_counts().sort_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=[f"{int(interval.left*100)}-{int(interval.right*100)}%" for interval in iv_counts.index],
            y=iv_counts.values,
            name='IV Distribution'
        ))

        fig3.update_layout(
            title='Distribution of Implied Volatility (IV)',
            xaxis_title='Implied Volatility Range',
            yaxis_title='Frequency',
            height=800
        )
        fig3.write_html(self.results_dir / "iv_distribution.html")

def main():
    """Example usage"""
    # Initialize analyzer for options
    end_date = date.today()
    start_date = end_date - timedelta(days=7)  # Get 1 week of data

    # Specify output path if needed
    output_path = None  # Change this to a specific path if desired

    options_analyzer = OptionsAnalyzer('BTC', start_date, end_date, output_path=output_path)

    try:
        # Fetch and process data
        data = options_analyzer.fetch_data()
        print(f"Retrieved {len(data)} records")

        # Generate all plots
        vol_surface = options_analyzer.plot_volatility_surface()
        greek_surfaces = options_analyzer.plot_greeks_3d()
        greeks_vs_moneyness = options_analyzer.plot_greeks_vs_moneyness()
        greeks_vs_iv = options_analyzer.plot_greeks_vs_iv()
        cp_ratio = options_analyzer.plot_cp_ratio()

        # Save plots
        vol_surface.write_html(options_analyzer.results_dir / "volatility_surface.html")
        for i, fig in enumerate(greek_surfaces):
            fig.write_html(options_analyzer.results_dir / f"greek_surface_{i+1}.html")
        greeks_vs_moneyness.write_html(options_analyzer.results_dir / "greeks_vs_moneyness.html")
        greeks_vs_iv.write_html(options_analyzer.results_dir / "greeks_vs_iv.html")
        cp_ratio.write_html(options_analyzer.results_dir / "cp_ratio.html")

        print("Analysis complete. Check the generated HTML files for interactive plots.")

        # Initialize option data analyzer
        option_data_analyzer = OptionDataAnalyzer(days_back=7, output_path=output_path)

        print("Fetching and processing option data...")
        option_data = option_data_analyzer.get_option_data()

        if not option_data.empty:
            output_file = options_analyzer.results_dir / f"Bitcoin_Call_Options_{option_data_analyzer.start_date}_{option_data_analyzer.end_date}.csv"
            option_data.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")

            print("Creating interactive visualizations...")
            option_data_analyzer.create_visualizations(option_data)
            print("Interactive visualizations saved as HTML files")
        else:
            print("No data retrieved for the specified date range.")

    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()