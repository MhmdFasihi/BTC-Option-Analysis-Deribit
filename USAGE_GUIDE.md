# BTC Options Analysis Dashboard - Usage Guide

## Quick Start

### 1. Environment Setup

```bash
# Activate the conda environment
conda activate crypto-option

# Navigate to project directory
cd /Users/mhmdfasihi/Desktop/Code/options/options-analysis
```

### 2. Launch Dashboard

```bash
# Run the Streamlit app
streamlit run app/main.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## Dashboard Overview

The dashboard consists of 5 main sections:

### 🏠 Main Page (Home)
**Purpose**: Data fetching, configuration, and analysis execution

**Features**:
- Currency selection (BTC, ETH)
- Date range picker
- Risk-free rate configuration
- Cache management
- Analysis execution button

**Workflow**:
1. Select currency (BTC or ETH)
2. Choose date range (max 365 days)
3. Optionally adjust risk-free rate
4. Click "🚀 Run Analysis"
5. Wait for data fetching and calculations
6. Navigate to analysis pages

**Important**: You must run analysis from this page before viewing other sections!

---

### ⚡ Gamma Exposure Page
**Purpose**: Understand market maker positioning and gamma squeeze risk

**4 Tabs**:

#### Tab 1: GEX Overview
- Current spot price and total GEX metrics
- Dual chart: Calls/Puts separately + Net GEX
- Gamma flip point marker
- Color coding:
  - Green: Call GEX (positive, stabilizing)
  - Red: Put GEX (negative, destabilizing)
  - Yellow: Current spot price
  - Cyan: Gamma flip point

#### Tab 2: Strike Analysis
- Filter by strike range
- Top 20 strikes by absolute GEX
- Distance from spot percentage
- Download CSV functionality

#### Tab 3: Gamma Squeeze
- 🚨 Squeeze risk indicator
- Pressure score gauge (0-100)
- Dealer gamma position
- Flip point distance
- Score interpretation:
  - 0-40: Low risk (green)
  - 40-70: Moderate risk (yellow)
  - 70-100: High risk (red)

#### Tab 4: Market Dynamics
- Price movement scenarios
- Market maker flow predictions (buy/sell)
- Pin risk analysis
- Effect indicators (stabilizing/destabilizing)

**Key Insights**:
- Positive GEX = Market makers provide support/resistance
- Negative GEX = Potential for explosive moves
- Gamma flip point = Critical level where dynamics change

---

### 📊 Overview Page
**Purpose**: Market summary and volume analysis

**Sections**:

#### Market Summary
- Current price
- Total volume (BTC and USD)
- Call/Put ratio
- Total contracts
- Average IV for calls and puts

#### Volume Analysis
- Daily volume chart (stacked calls/puts)
- Call/Put ratio time series
- Sentiment indicators (bullish/bearish)

#### Most Active Strikes
- Top 10 call strikes by volume
- Top 10 put strikes by volume
- Distance from spot
- Average IV
- Color-coded heat maps

#### Distribution Analysis
- Volume by maturity date
- Number of active strikes
- Time to maturity distribution (0-7d, 7-14d, etc.)
- Moneyness distribution (OTM, ATM, ITM)

**Trading Insights**:
- C/P ratio > 1.0 = Bullish sentiment
- C/P ratio < 1.0 = Bearish sentiment
- High volume at specific strikes = Key levels

---

### 📈 Volatility Surface Page
**Purpose**: Understand volatility expectations and skew

**4 Tabs**:

#### Tab 1: 3D Surface
- Interactive 3D visualization
- Select metric: IV, Vega, or Gamma
- Option type: Calls, Puts, or Both
- Spot price marker in red
- Rotate and zoom for different perspectives

#### Tab 2: IV Skew
- Separate charts for calls and puts
- Multiple maturity ranges (selectable)
- ATM marker (yellow line)
- **Patterns**:
  - Smile: Higher IV for OTM options
  - Smirk: Asymmetric skew
  - Flat: Similar IV across strikes

#### Tab 3: Term Structure
- ATM volatility by maturity
- Bubble size = volume
- **Patterns**:
  - Upward sloping = Normal (future uncertainty)
  - Downward sloping = Event-driven (near-term risk)
  - Humped = Specific event expected

#### Tab 4: Analysis
- Call vs Put IV spread
- IV distribution histogram
- Volatility percentiles (10%, 25%, 50%, 75%, 90%)
- Statistical summary

**Key Metrics**:
- Steepening skew = Increased fear
- Flattening skew = Market calming
- Backwardation = Near-term event risk

---

### 🎲 Greeks Analysis Page
**Purpose**: Portfolio risk and option sensitivities

**5 Tabs**:

#### Tab 1: Portfolio Greeks
- Total Delta, Gamma, Vega, Theta
- Dollar exposure for each Greek
- Breakdown by option type (calls vs puts)
- Greeks distribution by maturity

#### Tab 2: Delta Analysis
- Delta heatmap by strike and maturity
- Delta vs Moneyness curve
- **Interpretation**:
  - Call delta: 0 to 1
  - Put delta: -1 to 0
  - ATM delta ≈ ±0.5

#### Tab 3: Gamma Analysis
- Gamma heatmap
- Gamma profile near ATM (±20%)
- **Key Points**:
  - Always positive
  - Highest at ATM
  - Increases near expiration
  - Gamma scalping opportunities

#### Tab 4: Vega Analysis
- Vega heatmap
- Vega vs IV scatter (colored by TTM)
- **Insights**:
  - Higher for longer-dated options
  - Maximum at ATM
  - Critical in high-vol crypto markets

#### Tab 5: Theta Analysis
- Theta heatmap
- Theta decay curve vs TTM
- **Characteristics**:
  - Negative for option buyers
  - Accelerates near expiration
  - Non-linear decay curve

**Risk Management**:
- Monitor daily for concentration risk
- Delta hedge to neutralize directional exposure
- Gamma scalp for profit in volatile markets
- Manage vega exposure during vol regime changes

---

## Best Practices

### Data Fetching
1. **Start with small date ranges** (7 days) for testing
2. **Use cache** for repeated analysis (faster loading)
3. **Clear cache** if data seems stale
4. **Parallel workers**: Keep at 5 (optimal balance)

### Analysis Workflow
1. **Run analysis** from main page first
2. **Check Overview** to understand market conditions
3. **Gamma Exposure** for squeeze risk
4. **Volatility Surface** for IV dynamics
5. **Greeks** for detailed risk metrics

### Performance Tips
- Smaller date ranges = faster loading
- Use cache when possible
- Close unused browser tabs
- Refresh if dashboard becomes slow

### Interpretation Guide
1. **Start macro**: Overview page for market context
2. **Risk assessment**: Gamma squeeze indicators
3. **Volatility regime**: Check IV surface and term structure
4. **Position risk**: Analyze Greeks for specific exposures

---

## Troubleshooting

### Dashboard won't load
```bash
# Check environment
conda activate crypto-option

# Verify streamlit is installed
streamlit --version

# Reinstall if needed
pip install streamlit --upgrade
```

### No data displayed
- Ensure you clicked "🚀 Run Analysis" on main page
- Check date range (max 365 days)
- Verify internet connection (API access)
- Try clearing cache and re-running

### Slow performance
- Reduce date range
- Use cache for repeated queries
- Close other browser tabs
- Restart Streamlit server

### API errors
- Check internet connection
- Deribit API might be rate-limiting
- Wait a few minutes and retry
- Use cached data if available

### Missing visualizations
- Ensure data was fetched successfully
- Some charts require minimum data points
- Check browser console for errors
- Try refreshing the page

---

## Advanced Features

### Downloading Data
- Most tables have "Download CSV" buttons
- Data includes all calculated metrics
- Use for further analysis in Excel/Python

### Custom Analysis
- Adjust risk-free rate for different scenarios
- Filter by maturity buckets
- Focus on specific strike ranges
- Compare multiple time periods

### Session State
- Dashboard maintains state across pages
- No need to re-fetch data when navigating
- Clear cache to reset

---

## Keyboard Shortcuts

- **R**: Rerun the current page
- **C**: Clear cache
- **S**: Show sidebar (if hidden)
- **⌘/Ctrl + Enter**: Run code cells (if in development mode)

---

## Data Sources

- **Options Data**: Deribit API (historical trades)
- **Greeks**: Calculated using Black-76 model
- **GEX**: Computed from gamma, volume, and spot price
- **Historical Volatility**: WebSocket + REST API fallback

---

## Support

### Issues
- Report bugs on GitHub: https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit/issues
- Include screenshots and error messages
- Provide date range and settings used

### Documentation
- README.md: Project overview
- COMPREHENSIVE_PIPELINE.md: Development pipeline
- DETAILED_ROADMAP.md: Implementation guide

### Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
conda env update -f environment.yml
```

---

## Tips for Effective Analysis

1. **Start Broad**: Overview page → drill down to specifics
2. **Look for Divergences**: C/P ratio vs price action
3. **Monitor Gamma Flip**: Price crossing this level changes dynamics
4. **Watch Skew**: Changes in skew precede volatility regime shifts
5. **Track Greeks Daily**: Portfolio risk changes with market moves
6. **Use Multiple Timeframes**: Compare 7d, 30d, 90d perspectives

---

## Example Workflow

### Daily Analysis Routine
```
1. Open dashboard
2. Select BTC, last 7 days
3. Run analysis
4. Check Overview for market sentiment (C/P ratio)
5. Review Gamma Exposure for squeeze risk
6. Monitor Volatility Surface for IV changes
7. Assess Greeks for portfolio risk
8. Export data for records
```

### Pre-Trade Analysis
```
1. Select relevant time period
2. Identify support/resistance from GEX
3. Check IV skew for option pricing
4. Calculate expected Greeks for planned position
5. Assess gamma squeeze risk
6. Review term structure for timing
```

### Risk Management Check
```
1. Portfolio Greeks tab → overall exposure
2. Gamma Analysis → rehedging frequency
3. Vega Analysis → vol risk
4. Theta Analysis → time decay
5. Delta Analysis → directional bias
```

---

**Last Updated**: December 11, 2025
**Version**: 1.0.0
**Python**: 3.11
**Streamlit**: 1.28+
