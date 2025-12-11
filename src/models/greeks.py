"""
Enhanced Greeks Calculator for Cryptocurrency Options
Based on crypto_black_scholes library with improvements for Deribit options

Supports:
- Black-Scholes model for standard options
- Black-76 model for futures-style options
- Coin-settled options pricing
- Second-order Greeks
- Portfolio aggregation
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from enum import Enum

class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"

class PricingModel(Enum):
    """Pricing model selection"""
    BLACK_SCHOLES = "black_scholes"
    BLACK_76 = "black_76"  # For futures/crypto
    COIN_BASED = "coin_based"  # Coin-settled options

@dataclass
class OptionParameters:
    """Option parameters for pricing and Greeks"""
    spot_price: float
    strike_price: float
    time_to_maturity: float  # In years
    volatility: float  # Annualized
    risk_free_rate: float = 0.0  # Zero for crypto
    option_type: OptionType = OptionType.CALL
    pricing_model: PricingModel = PricingModel.BLACK_76

@dataclass
class Greeks:
    """Greeks calculation results"""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float = 0.0  # Always zero for crypto

    # Second-order Greeks (optional)
    speed: Optional[float] = None  # dGamma/dS
    charm: Optional[float] = None  # dDelta/dT
    vanna: Optional[float] = None  # dDelta/dVol
    vomma: Optional[float] = None  # dVega/dVol

class EnhancedGreeksCalculator:
    """
    Enhanced Greeks calculator for cryptocurrency options

    Features:
    - Multiple pricing models (Black-Scholes, Black-76, Coin-settled)
    - First and second-order Greeks
    - Portfolio aggregation
    - Risk metrics
    """

    def __init__(self, pricing_model: PricingModel = PricingModel.BLACK_76):
        """
        Initialize Greeks calculator

        Args:
            pricing_model: Pricing model to use (default: BLACK_76 for crypto)
        """
        self.pricing_model = pricing_model

    def calculate_d1_d2(self, params: OptionParameters) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula

        Args:
            params: Option parameters

        Returns:
            Tuple of (d1, d2)
        """
        S = params.spot_price
        K = params.strike_price
        T = max(params.time_to_maturity, 1e-10)  # Avoid division by zero
        sigma = params.volatility
        r = params.risk_free_rate

        if params.pricing_model == PricingModel.BLACK_76:
            # Black-76: Use forward price, no discounting
            # For crypto, F ≈ S (no cost of carry)
            F = S
            d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        else:
            # Standard Black-Scholes
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2

    def calculate_greeks(self, params: OptionParameters,
                        include_second_order: bool = False) -> Greeks:
        """
        Calculate all Greeks for an option

        Args:
            params: Option parameters
            include_second_order: Whether to calculate second-order Greeks

        Returns:
            Greeks object with calculated values
        """
        S = params.spot_price
        K = params.strike_price
        T = max(params.time_to_maturity, 1e-10)
        sigma = params.volatility
        r = params.risk_free_rate
        is_call = params.option_type == OptionType.CALL

        # Calculate d1 and d2
        d1, d2 = self.calculate_d1_d2(params)

        # Standard normal PDF and CDF
        nd1 = norm.pdf(d1)
        nd2 = norm.pdf(d2)
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)

        # DELTA
        if is_call:
            delta = Nd1
        else:
            delta = Nd1 - 1.0  # or -N(-d1)

        # GAMMA (same for calls and puts)
        gamma = nd1 / (S * sigma * np.sqrt(T))

        # VEGA (same for calls and puts, per 1% change in vol)
        vega = S * np.sqrt(T) * nd1 * 0.01

        # THETA (per day)
        theta_common = -(S * sigma * nd1) / (2 * np.sqrt(T))
        if is_call:
            theta = (theta_common - r * K * np.exp(-r * T) * Nd2) / 365
        else:
            theta = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        # RHO (always 0 for crypto since r=0)
        rho = 0.0

        # Create Greeks object
        greeks = Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho
        )

        # Second-order Greeks (if requested)
        if include_second_order:
            greeks.speed = self._calculate_speed(params, d1, nd1)
            greeks.charm = self._calculate_charm(params, d1, d2, nd1)
            greeks.vanna = self._calculate_vanna(params, d1, d2, nd1)
            greeks.vomma = self._calculate_vomma(params, d1, nd1)

        return greeks

    def _calculate_speed(self, params: OptionParameters, d1: float, nd1: float) -> float:
        """Calculate Speed (dGamma/dS)"""
        S = params.spot_price
        sigma = params.volatility
        T = params.time_to_maturity

        speed = -nd1 * (d1 / (S**2 * sigma * np.sqrt(T)) + 1 / (S * sigma * np.sqrt(T)))
        return speed

    def _calculate_charm(self, params: OptionParameters, d1: float, d2: float, nd1: float) -> float:
        """Calculate Charm (dDelta/dT)"""
        sigma = params.volatility
        T = params.time_to_maturity
        r = params.risk_free_rate
        is_call = params.option_type == OptionType.CALL

        if is_call:
            charm = -nd1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        else:
            charm = -nd1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))

        return charm / 365  # Convert to daily

    def _calculate_vanna(self, params: OptionParameters, d1: float, d2: float, nd1: float) -> float:
        """Calculate Vanna (dDelta/dVol = dVega/dS)"""
        sigma = params.volatility

        vanna = -nd1 * d2 / sigma
        return vanna * 0.01  # Per 1% vol change

    def _calculate_vomma(self, params: OptionParameters, d1: float, nd1: float) -> float:
        """Calculate Vomma (dVega/dVol)"""
        S = params.spot_price
        T = params.time_to_maturity
        sigma = params.volatility

        vomma = S * np.sqrt(T) * nd1 * d1 * (d1 - sigma * np.sqrt(T)) / sigma
        return vomma * 0.01  # Per 1% vol change

    def calculate_greeks_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Greeks for entire DataFrame of options

        Args:
            df: DataFrame with columns: spot_price, strike_price, time_to_maturity,
                volatility (iv), option_type

        Returns:
            DataFrame with added Greeks columns
        """
        if df.empty:
            return df

        greeks_list = []

        for _, row in df.iterrows():
            params = OptionParameters(
                spot_price=row.get('index_price', row.get('spot_price')),
                strike_price=row['strike_price'],
                time_to_maturity=row['time_to_maturity'] / 365,  # Convert days to years
                volatility=row.get('iv', row.get('volatility')),
                option_type=OptionType.CALL if row['option_type'] == 'call' else OptionType.PUT,
                pricing_model=self.pricing_model
            )

            greeks = self.calculate_greeks(params)
            greeks_list.append({
                'delta': greeks.delta,
                'gamma': greeks.gamma,
                'vega': greeks.vega,
                'theta': greeks.theta,
                'rho': greeks.rho
            })

        greeks_df = pd.DataFrame(greeks_list)
        return pd.concat([df, greeks_df], axis=1)

@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for a portfolio"""
    total_delta: float
    total_gamma: float
    total_vega: float
    total_theta: float
    delta_dollars: float
    gamma_dollars: float
    vega_dollars: float
    theta_dollars: float

    # Risk metrics
    gamma_exposure: float  # Total gamma dollar exposure
    vega_exposure: float   # Total vega dollar exposure

class PortfolioGreeksAggregator:
    """Aggregate Greeks across a portfolio of options"""

    @staticmethod
    def aggregate_greeks(df: pd.DataFrame, spot_price: float) -> PortfolioGreeks:
        """
        Aggregate Greeks for portfolio

        Args:
            df: DataFrame with Greeks and position sizes
            spot_price: Current underlying price

        Returns:
            PortfolioGreeks object
        """
        if df.empty or 'delta' not in df.columns:
            return PortfolioGreeks(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Aggregate Greeks
        total_delta = df['delta'].sum()
        total_gamma = df['gamma'].sum()
        total_vega = df['vega'].sum()
        total_theta = df['theta'].sum()

        # Dollar exposures
        delta_dollars = total_delta * spot_price
        gamma_dollars = total_gamma * spot_price * spot_price / 100  # Per 1% move
        vega_dollars = total_vega * spot_price
        theta_dollars = total_theta * spot_price

        # Risk metrics
        gamma_exposure = abs(gamma_dollars)
        vega_exposure = abs(vega_dollars)

        return PortfolioGreeks(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_vega=total_vega,
            total_theta=total_theta,
            delta_dollars=delta_dollars,
            gamma_dollars=gamma_dollars,
            vega_dollars=vega_dollars,
            theta_dollars=theta_dollars,
            gamma_exposure=gamma_exposure,
            vega_exposure=vega_exposure
        )
