"""
Data Collection Module for Deribit Options
Enhanced with caching, parallel processing, and progress tracking
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime as dt, date, timedelta
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging
import asyncio
import websockets
import json

logger = logging.getLogger(__name__)


def datetime_to_timestamp(datetime_obj: dt) -> int:
    """Convert datetime to millisecond timestamp"""
    return int(dt.timestamp(datetime_obj) * 1000)


class OptionsDataFetcher:
    """
    Fetch options data from Deribit API with caching and parallel processing

    Features:
    - Parallel data fetching for faster collection
    - Intelligent file-based caching
    - Automatic retry with exponential backoff
    - Progress tracking with tqdm
    - WebSocket + REST API fallback for historical volatility
    """

    def __init__(self,
                 currencies: Union[List[str], str],
                 start_date: date,
                 end_date: date,
                 cache_dir: str = "cache",
                 max_workers: int = 5):
        """
        Initialize data fetcher

        Args:
            currencies: Single currency or list of currencies (BTC, ETH)
            start_date: Start date for data collection
            end_date: End date for data collection
            cache_dir: Directory for caching data
            max_workers: Number of parallel workers for fetching
        """
        self.currencies = [currencies.upper()] if isinstance(currencies, str) else [c.upper() for c in currencies]
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.max_workers = max_workers
        self.session = requests.Session()
        self._validate_inputs()

        logger.info(f"OptionsDataFetcher initialized for {self.currencies}")

    def _validate_inputs(self) -> None:
        """Validate input parameters"""
        for currency in self.currencies:
            if currency not in ['BTC', 'ETH']:
                raise ValueError(f"Currency {currency} is not supported. Choose from BTC, ETH")
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
        if (self.end_date - self.start_date).days > 365:
            raise ValueError("Date range cannot exceed 365 days")

    def _get_cache_filename(self, currency: str) -> Path:
        """Get cache filename for currency and date range"""
        return self.cache_dir / f"{currency}_options_{self.start_date}_{self.end_date}.csv"

    def _fetch_trades_chunk(self, currency: str, start_ts: int, end_ts: int) -> List[Dict]:
        """
        Fetch a chunk of trades from Deribit API

        Args:
            currency: Currency to fetch
            start_ts: Start timestamp (ms)
            end_ts: End timestamp (ms)

        Returns:
            List of trade dictionaries
        """
        params = {
            "currency": currency,
            "kind": "option",
            "count": 10000,
            "include_old": True,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts
        }

        url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'

        # Retry with exponential backoff
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
                    logger.error(f"Failed to get data for {currency} from {start_ts} to {end_ts}: {e}")
                    return []
                time.sleep(0.5 * (2 ** attempt))
        return []

    def _parse_instrument_name(self, name: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """
        Parse Deribit instrument name
        Format: BTC-25DEC25-100000-C

        Args:
            name: Instrument name string

        Returns:
            Tuple of (maturity, strike, option_type)
        """
        try:
            parts = name.split('-')
            maturity = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            return maturity, strike, option_type
        except Exception as e:
            logger.warning(f"Error parsing instrument name {name}: {e}")
            return None, None, None

    def _process_chunk(self, chunk: List[Dict]) -> pd.DataFrame:
        """
        Process raw trades data into structured DataFrame

        Args:
            chunk: List of trade dictionaries

        Returns:
            Processed DataFrame
        """
        if not chunk:
            return pd.DataFrame()

        df = pd.DataFrame(chunk)

        # Parse instrument names
        instrument_details = [self._parse_instrument_name(name) for name in df['instrument_name']]
        df['maturity_date'] = [details[0] for details in instrument_details]
        df['strike_price'] = [details[1] for details in instrument_details]
        df['option_type'] = [details[2] for details in instrument_details]

        # Convert timestamps and dates
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['maturity_date'] = pd.to_datetime(df['maturity_date'], format='%d%b%y')

        # Create derived columns
        df['date'] = df['date_time'].dt.date
        df['time_to_maturity'] = (df['maturity_date'] - df['date_time']).dt.total_seconds() / 86400
        df['moneyness'] = df['index_price'] / df['strike_price']
        df['iv'] = df['iv'] / 100  # Convert from percentage
        df['is_call'] = df['option_type'] == 'call'

        # Calculate volume metrics
        df['volume_btc'] = df['price'] * df['contracts']
        df['volume_usd'] = df['volume_btc'] * df['index_price']

        return df

    def fetch_data(self, currency: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch options data for a single currency

        Args:
            currency: Currency to fetch (BTC or ETH)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with options data
        """
        cache_file = self._get_cache_filename(currency)

        # Check cache
        if use_cache and cache_file.exists():
            logger.info(f"Loading {currency} data from cache...")
            try:
                return pd.read_csv(cache_file, parse_dates=['date_time', 'maturity_date'])
            except Exception as e:
                logger.warning(f"Error loading cache: {e}. Fetching fresh data...")

        logger.info(f"Fetching {currency} data from API...")

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

        # Parallel fetching
        all_trades = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._fetch_trades_chunk, currency, start_ts, end_ts)
                for start_ts, end_ts in date_chunks
            ]

            for future in tqdm(as_completed(futures), total=len(futures),
                             desc=f"Fetching {currency} data", unit="day"):
                trades = future.result()
                if trades:
                    all_trades.extend(trades)

        if not all_trades:
            logger.warning(f"No data retrieved for {currency}")
            return pd.DataFrame()

        # Process and cache
        df = self._process_chunk(all_trades)
        df.to_csv(cache_file, index=False)
        logger.info(f"Saved {len(df)} rows to cache for {currency}")

        return df

    def fetch_all_data(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all currencies

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary mapping currency to DataFrame
        """
        results = {}
        for currency in self.currencies:
            results[currency] = self.fetch_data(currency, use_cache)
        return results

    async def _fetch_historical_volatility_ws(self, currency: str) -> Optional[List[Dict]]:
        """
        Fetch historical volatility from Deribit WebSocket API

        Args:
            currency: Currency (BTC or ETH)

        Returns:
            List of historical volatility data points
        """
        try:
            msg = {
                "jsonrpc": "2.0",
                "id": 8387,
                "method": "public/get_historical_volatility",
                "params": {"currency": currency}
            }

            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as websocket:
                await websocket.send(json.dumps(msg))
                response = await websocket.recv()
                data = json.loads(response)

                if "result" in data:
                    logger.info(f"Fetched historical volatility for {currency} via WebSocket")
                    return data["result"]

        except Exception as e:
            logger.warning(f"WebSocket fetch failed for {currency}: {e}")

        return None

    def fetch_historical_volatility(self, currency: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical volatility with WebSocket + REST fallback

        Args:
            currency: Currency (BTC or ETH)

        Returns:
            DataFrame with historical volatility data
        """
        # Try WebSocket first
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            hv_data = loop.run_until_complete(self._fetch_historical_volatility_ws(currency))
            if hv_data:
                return pd.DataFrame(hv_data)
        except Exception as e:
            logger.warning(f"Failed to fetch historical volatility: {e}")

        # REST API fallback
        try:
            url = 'https://www.deribit.com/api/v2/public/get_historical_volatility'
            params = {"currency": currency}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "result" in data:
                logger.info(f"Fetched historical volatility for {currency} via REST API")
                return pd.DataFrame(data["result"])

        except Exception as e:
            logger.error(f"REST API fallback failed: {e}")

        return None

    def clear_cache(self, currency: Optional[str] = None):
        """
        Clear cached data

        Args:
            currency: Specific currency to clear, or None for all
        """
        if currency:
            cache_file = self._get_cache_filename(currency)
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache for {currency}")
        else:
            for file in self.cache_dir.glob("*.csv"):
                file.unlink()
            logger.info("Cleared all cache files")
