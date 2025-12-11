"""
Gamma Exposure (GEX) and Gamma Squeeze Analysis
CRITICAL MODULE for market maker positioning and gamma squeeze detection

Features:
- Gamma Exposure by strike price
- Gamma squeeze detection
- Net GEX profiles
- GEX flip points (gamma neutral levels)
- Pin risk analysis
- Market maker positioning inference
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GammaExposureMetrics:
    """Gamma exposure metrics for a strike or overall market"""
    strike_price: float
    call_gex: float
    put_gex: float
    net_gex: float
    total_gamma: float
    call_volume: float
    put_volume: float
    distance_from_spot: float  # Percentage


@dataclass
class GammaSqueezeIndicators:
    """Indicators for potential gamma squeeze"""
    is_gamma_squeeze_risk: bool
    net_dealer_gamma: float  # Negative = dealers short gamma
    largest_negative_gex_strike: Optional[float]
    largest_positive_gex_strike: Optional[float]
    gamma_flip_point: Optional[float]  # Price where net GEX = 0
    spot_to_flip_distance: Optional[float]  # % distance
    squeeze_pressure_score: float  # 0-100, higher = more squeeze risk


class GammaExposureAnalyzer:
    """
    Analyze Gamma Exposure (GEX) for options market

    Gamma Exposure represents the hedging flow that market makers must execute
    when the underlying price moves. Understanding GEX helps identify:
    - Support/resistance levels (high positive GEX)
    - Breakout zones (negative GEX)
    - Gamma squeeze potential
    - Market maker positioning
    """

    def __init__(self, data: pd.DataFrame, currency: str):
        """
        Initialize GEX analyzer

        Args:
            data: Options data with Greeks calculated
            currency: Currency being analyzed (BTC/ETH)
        """
        self.data = data
        self.currency = currency
        self.spot_price = data['index_price'].iloc[-1] if not data.empty else 0

        if data.empty:
            logger.warning(f"Empty data provided for {currency} GEX analysis")
        elif 'gamma' not in data.columns:
            logger.error(f"Gamma column not found in data for {currency}")

    def calculate_gamma_exposure_by_strike(self) -> pd.DataFrame:
        """
        Calculate Gamma Exposure (GEX) by strike price

        GEX Formula:
        GEX = Gamma * Open Interest * Contract Size * Spot Price

        For simplicity with trade data (no OI):
        GEX = Gamma * Volume * Spot Price

        Interpretation:
        - Positive GEX (calls): Dealers buy on rallies, sell on dips (stabilizing)
        - Negative GEX (puts): Dealers sell on rallies, buy on dips (destabilizing)

        Returns:
            DataFrame with GEX by strike
        """
        if self.data.empty or 'gamma' not in self.data.columns:
            logger.warning("Cannot calculate GEX - missing data or gamma")
            return pd.DataFrame()

        gex_data = []

        # Calculate GEX for calls and puts separately
        for option_type, sign in [('call', 1), ('put', -1)]:
            type_data = self.data[self.data['option_type'] == option_type]

            if type_data.empty:
                continue

            # Group by strike price
            strike_groups = type_data.groupby('strike_price')

            for strike, group in strike_groups:
                # Volume-weighted average gamma
                total_volume = group['volume_btc'].sum()
                avg_gamma = group['gamma'].mean()

                # Calculate GEX
                # Sign convention: calls positive, puts negative
                gex = sign * avg_gamma * total_volume * self.spot_price

                gex_data.append({
                    'strike_price': strike,
                    'option_type': option_type,
                    'gex': gex,
                    'volume_btc': total_volume,
                    'avg_gamma': avg_gamma,
                    'distance_from_spot_pct': ((strike - self.spot_price) / self.spot_price) * 100
                })

        gex_df = pd.DataFrame(gex_data)

        if not gex_df.empty:
            # Calculate net GEX by strike (combining calls and puts)
            net_gex = gex_df.groupby('strike_price')['gex'].sum().reset_index()
            net_gex['option_type'] = 'net'
            net_gex['volume_btc'] = 0
            net_gex['avg_gamma'] = 0
            net_gex['distance_from_spot_pct'] = ((net_gex['strike_price'] - self.spot_price) / self.spot_price) * 100

            # Combine with individual call/put data
            gex_df = pd.concat([gex_df, net_gex], ignore_index=True)

            logger.info(f"Calculated GEX for {len(gex_df[gex_df['option_type']=='net'])} strikes")

        return gex_df

    def detect_gamma_squeeze_risk(self, gex_df: pd.DataFrame) -> GammaSqueezeIndicators:
        """
        Detect potential gamma squeeze conditions

        Gamma Squeeze occurs when:
        1. Dealers are SHORT gamma (negative net GEX)
        2. Price moves through strikes with large negative GEX
        3. Dealers must buy into rising price (or sell into falling)

        Args:
            gex_df: GEX data by strike

        Returns:
            GammaSqueezeIndicators object
        """
        if gex_df.empty:
            return GammaSqueezeIndicators(
                is_gamma_squeeze_risk=False,
                net_dealer_gamma=0,
                largest_negative_gex_strike=None,
                largest_positive_gex_strike=None,
                gamma_flip_point=None,
                spot_to_flip_distance=None,
                squeeze_pressure_score=0
            )

        # Get net GEX data
        net_gex_df = gex_df[gex_df['option_type'] == 'net'].copy()

        if net_gex_df.empty:
            return GammaSqueezeIndicators(
                is_gamma_squeeze_risk=False,
                net_dealer_gamma=0,
                largest_negative_gex_strike=None,
                largest_positive_gex_strike=None,
                gamma_flip_point=None,
                spot_to_flip_distance=None,
                squeeze_pressure_score=0
            )

        # Total net dealer gamma (market makers take opposite side)
        net_dealer_gamma = -net_gex_df['gex'].sum()  # Opposite of market

        # Find largest negative and positive GEX strikes
        largest_negative_idx = net_gex_df['gex'].idxmin()
        largest_positive_idx = net_gex_df['gex'].idxmax()

        largest_negative_gex_strike = net_gex_df.loc[largest_negative_idx, 'strike_price']
        largest_positive_gex_strike = net_gex_df.loc[largest_positive_idx, 'strike_price']

        # Find gamma flip point (where net GEX crosses zero)
        gamma_flip_point = self._find_gamma_flip_point(net_gex_df)

        # Calculate distance from spot to flip point
        spot_to_flip_distance = None
        if gamma_flip_point:
            spot_to_flip_distance = ((gamma_flip_point - self.spot_price) / self.spot_price) * 100

        # Calculate squeeze pressure score (0-100)
        squeeze_pressure_score = self._calculate_squeeze_pressure(
            net_dealer_gamma,
            net_gex_df,
            gamma_flip_point
        )

        # Determine if there's squeeze risk
        is_gamma_squeeze_risk = (
            net_dealer_gamma < 0 and  # Dealers short gamma
            squeeze_pressure_score > 50  # High pressure
        )

        return GammaSqueezeIndicators(
            is_gamma_squeeze_risk=is_gamma_squeeze_risk,
            net_dealer_gamma=net_dealer_gamma,
            largest_negative_gex_strike=largest_negative_gex_strike,
            largest_positive_gex_strike=largest_positive_gex_strike,
            gamma_flip_point=gamma_flip_point,
            spot_to_flip_distance=spot_to_flip_distance,
            squeeze_pressure_score=squeeze_pressure_score
        )

    def _find_gamma_flip_point(self, net_gex_df: pd.DataFrame) -> Optional[float]:
        """
        Find the strike price where net GEX changes sign (crosses zero)

        This is the "gamma neutral" level - important for market dynamics

        Args:
            net_gex_df: Net GEX data

        Returns:
            Gamma flip point strike price, or None if not found
        """
        if net_gex_df.empty:
            return None

        # Sort by strike
        net_gex_df = net_gex_df.sort_values('strike_price')

        # Find sign changes
        sign_changes = np.diff(np.sign(net_gex_df['gex'].values))
        zero_crossings = np.where(sign_changes != 0)[0]

        if len(zero_crossings) == 0:
            return None

        # Find crossing closest to current spot price
        closest_idx = zero_crossings[0]
        closest_distance = abs(net_gex_df.iloc[closest_idx]['strike_price'] - self.spot_price)

        for idx in zero_crossings[1:]:
            distance = abs(net_gex_df.iloc[idx]['strike_price'] - self.spot_price)
            if distance < closest_distance:
                closest_idx = idx
                closest_distance = distance

        # Interpolate between the two strikes
        strike_below = net_gex_df.iloc[closest_idx]['strike_price']
        strike_above = net_gex_df.iloc[closest_idx + 1]['strike_price']
        gex_below = net_gex_df.iloc[closest_idx]['gex']
        gex_above = net_gex_df.iloc[closest_idx + 1]['gex']

        if gex_above == gex_below:
            return strike_below

        # Linear interpolation
        flip_point = strike_below + (strike_above - strike_below) * (-gex_below) / (gex_above - gex_below)

        return flip_point

    def _calculate_squeeze_pressure(self,
                                   net_dealer_gamma: float,
                                   net_gex_df: pd.DataFrame,
                                   gamma_flip_point: Optional[float]) -> float:
        """
        Calculate gamma squeeze pressure score (0-100)

        Factors:
        1. Magnitude of dealer short gamma
        2. Proximity to gamma flip point
        3. Concentration of negative GEX near spot
        4. Volume in short gamma strikes

        Args:
            net_dealer_gamma: Total net dealer gamma position
            net_gex_df: Net GEX data by strike
            gamma_flip_point: Gamma neutral price level

        Returns:
            Pressure score 0-100 (higher = more squeeze risk)
        """
        if net_gex_df.empty:
            return 0

        score = 0

        # Factor 1: Dealer short gamma magnitude (0-40 points)
        if net_dealer_gamma < 0:
            # Normalize by spot price
            normalized_short_gamma = abs(net_dealer_gamma) / (self.spot_price * 1000)
            score += min(40, normalized_short_gamma * 10)

        # Factor 2: Proximity to flip point (0-30 points)
        if gamma_flip_point:
            distance_pct = abs((gamma_flip_point - self.spot_price) / self.spot_price) * 100
            if distance_pct < 1:  # Very close (<1%)
                score += 30
            elif distance_pct < 3:  # Close (<3%)
                score += 20
            elif distance_pct < 5:  # Moderate (<5%)
                score += 10

        # Factor 3: Concentration of negative GEX near spot (0-30 points)
        # Look at GEX within +/- 10% of spot
        near_spot_df = net_gex_df[
            (net_gex_df['strike_price'] >= self.spot_price * 0.9) &
            (net_gex_df['strike_price'] <= self.spot_price * 1.1)
        ]

        if not near_spot_df.empty:
            negative_gex_near_spot = near_spot_df[near_spot_df['gex'] < 0]['gex'].sum()
            if abs(negative_gex_near_spot) > 0:
                # Normalize
                normalized_concentration = abs(negative_gex_near_spot) / (self.spot_price * 500)
                score += min(30, normalized_concentration * 10)

        return min(100, score)

    def analyze_pin_risk(self, gex_df: pd.DataFrame, threshold_percentile: float = 90) -> Dict:
        """
        Analyze pin risk - concentration of gamma at specific strikes

        Pin risk occurs when large gamma forces price to "pin" at a strike near expiration

        Args:
            gex_df: GEX data by strike
            threshold_percentile: Percentile for identifying high GEX strikes

        Returns:
            Dictionary with pin risk analysis
        """
        if gex_df.empty:
            return {'has_pin_risk': False, 'pin_strikes': []}

        net_gex_df = gex_df[gex_df['option_type'] == 'net'].copy()

        if net_gex_df.empty:
            return {'has_pin_risk': False, 'pin_strikes': []}

        # Find strikes with very high positive GEX (pin risk)
        threshold = np.percentile(net_gex_df['gex'], threshold_percentile)
        pin_strikes = net_gex_df[net_gex_df['gex'] > threshold]['strike_price'].tolist()

        # Check if any pin strikes are near current price (+/- 5%)
        near_price_pins = [
            strike for strike in pin_strikes
            if abs((strike - self.spot_price) / self.spot_price) < 0.05
        ]

        has_pin_risk = len(near_price_pins) > 0

        return {
            'has_pin_risk': has_pin_risk,
            'pin_strikes': pin_strikes,
            'near_price_pins': near_price_pins,
            'threshold_gex': threshold
        }

    def get_gex_profile(self, gex_df: pd.DataFrame,
                       strike_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Get GEX profile for visualization

        Args:
            gex_df: GEX data
            strike_range: Optional (min_strike, max_strike) to filter

        Returns:
            DataFrame ready for plotting
        """
        if gex_df.empty:
            return pd.DataFrame()

        profile_df = gex_df.copy()

        if strike_range:
            profile_df = profile_df[
                (profile_df['strike_price'] >= strike_range[0]) &
                (profile_df['strike_price'] <= strike_range[1])
            ]

        return profile_df.sort_values('strike_price')
