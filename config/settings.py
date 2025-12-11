"""
Configuration settings for BTC Options Analysis Dashboard
"""

from pathlib import Path
from typing import List

class Config:
    """Application configuration"""

    # API Configuration
    DERIBIT_API_URL = "https://history.deribit.com/api/v2"
    DERIBIT_WS_URL = "wss://www.deribit.com/ws/api/v2"
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

    # Data Configuration
    CACHE_DIR = Path("cache")
    OUTPUT_DIR = Path("Option_Analysis_Results")
    MAX_WORKERS = 5
    REQUEST_TIMEOUT = 10
    MAX_DATE_RANGE_DAYS = 365
    CACHE_TTL_SECONDS = 300  # 5 minutes

    # Analysis Configuration
    DEFAULT_RISK_FREE_RATE = 0.03
    SUPPORTED_CURRENCIES: List[str] = ["BTC", "ETH"]

    # Greeks Calculation
    GREEKS_PRECISION = 1e-6
    IV_SOLVER_TOLERANCE = 1e-8
    IV_SOLVER_MAX_ITERATIONS = 100

    # Visualization Configuration
    PLOTLY_THEME = "plotly_dark"
    DEFAULT_CHART_HEIGHT = 800
    DEFAULT_CHART_WIDTH = 1200
    COLOR_SCHEME_CALLS = "#00CC96"  # Green
    COLOR_SCHEME_PUTS = "#EF553B"   # Red
    COLOR_SCHEME_NEUTRAL = "#636EFA"  # Blue

    # Gamma Exposure
    GEX_SPOT_RANGE = (0.7, 1.3)  # 70% to 130% of current price
    GEX_VOL_RANGE = (-0.5, 1.0)   # -50% to +100% vol change
    GEX_GRID_POINTS = 50

    # Dashboard
    REFRESH_INTERVAL_SECONDS = 60
    MAX_DISPLAY_ROWS = 1000

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        cls.CACHE_DIR.mkdir(exist_ok=True, parents=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
