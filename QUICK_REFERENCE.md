# BTC Options Dashboard - Quick Reference Card

## 🚀 Launch Dashboard
```bash
conda activate crypto-option
cd /Users/mhmdfasihi/Desktop/Code/options/options-analysis
streamlit run app/main.py
```
Opens at: `http://localhost:8501`

---

## 📋 Daily Workflow

### 1. Setup (Main Page)
1. ✅ Select currency (BTC/ETH)
2. ✅ Choose date range (7-30 days recommended)
3. ✅ Click "🚀 Run Analysis"
4. ✅ Wait for data fetching (~30 seconds)

### 2. Risk Check (Gamma Exposure Page)
- **Squeeze Risk Gauge**:
  - 🟢 0-40: Low risk
  - 🟡 40-70: Moderate risk
  - 🔴 70-100: High risk
- **Gamma Flip Point**: Critical support/resistance level
- **Net GEX Chart**: Positive = stabilizing, Negative = explosive

### 3. Market Context (Overview Page)
- **C/P Ratio**:
  - > 1.0 = Bullish sentiment
  - < 1.0 = Bearish sentiment
- **Most Active Strikes**: Key price levels
- **Volume Trends**: Flow analysis

### 4. Volatility Check (Volatility Surface)
- **3D Surface**: Overall IV landscape
- **IV Skew**: Fear gauge (steepening = more fear)
- **Term Structure**: Near-term vs long-term expectations

### 5. Portfolio Risk (Greeks Analysis)
- **Delta**: Directional exposure
- **Gamma**: Rehedging frequency
- **Vega**: Volatility risk
- **Theta**: Time decay

---

## 🎯 Key Metrics Explained

### Gamma Exposure (GEX)
- **Positive GEX**: Market makers stabilize price (dampens volatility)
- **Negative GEX**: Market makers amplify price moves (increases volatility)
- **Flip Point**: Where GEX changes from positive to negative

### Greeks Quick Guide
| Greek | Measures | Sign | Max Value |
|-------|----------|------|-----------|
| Delta | Price sensitivity | ±1 | ATM options |
| Gamma | Delta change rate | Always + | ATM, near expiry |
| Vega | Vol sensitivity | Always + | ATM, longer dated |
| Theta | Time decay | Always - | ATM, near expiry |

### IV Skew Patterns
- **Smile**: Higher IV for OTM options (both sides)
- **Smirk**: Asymmetric (put skew in crypto)
- **Flat**: Similar IV across strikes

---

## ⚡ Common Tasks

### Export Data
- Click "Download CSV" on any table
- Use for further analysis in Excel/Python

### Clear Cache
- Sidebar → Advanced Options → Clear Cache
- Use if data seems stale

### Change Settings
- Sidebar → Risk-Free Rate (usually keep at 0%)
- Sidebar → Parallel Workers (keep at 5)

### Navigate Pages
- Sidebar menu
- No need to re-run analysis when switching pages

---

## 🔍 Trading Insights

### Support/Resistance Levels
1. Check **Gamma Flip Point** (GEX page)
2. Look for **high volume strikes** (Overview page)
3. Analyze **IV skew** for option pricing (Vol Surface page)

### Squeeze Risk Assessment
1. **Squeeze Pressure Score** > 70 = High risk
2. **Spot near Flip Point** = Critical zone
3. **Negative Net GEX** = Potential explosive move

### Volatility Regime
1. **Upward sloping term structure** = Normal
2. **Downward sloping** = Event-driven
3. **Steepening skew** = Increasing fear

### Position Risk
1. **Check Portfolio Greeks** (Tab 1)
2. **Monitor daily changes** in Delta/Gamma
3. **Assess Vega exposure** during vol regime changes

---

## ⚠️ Troubleshooting

### Dashboard won't load
```bash
conda activate crypto-option
streamlit --version  # Verify installation
streamlit run app/main.py
```

### No data displayed
- Did you click "Run Analysis"?
- Check date range (max 365 days)
- Verify internet connection

### Slow performance
- Reduce date range to 7-14 days
- Use cache (checkbox in sidebar)
- Restart Streamlit (Ctrl+C, then rerun)

### API errors
- Wait 2-3 minutes (rate limiting)
- Use cached data
- Try smaller date range

---

## 📊 Page-by-Page Guide

### Main Page (Home)
**Purpose**: Configure and fetch data
**Action**: Select settings → Run Analysis
**Next**: Navigate to analysis pages

### Page 1: ⚡ Gamma Exposure
**Purpose**: Understand market maker positioning
**Key Metrics**: Squeeze score, flip point, net GEX
**Action**: Assess risk before trading

### Page 2: 📊 Overview
**Purpose**: Market context and flow
**Key Metrics**: C/P ratio, volume, active strikes
**Action**: Identify sentiment and key levels

### Page 3: 📈 Volatility Surface
**Purpose**: IV dynamics and pricing
**Key Metrics**: 3D surface, skew, term structure
**Action**: Evaluate option pricing and volatility regime

### Page 4: 🎲 Greeks Analysis
**Purpose**: Portfolio risk management
**Key Metrics**: Delta, Gamma, Vega, Theta
**Action**: Monitor sensitivity exposures

---

## 💡 Pro Tips

1. **Start with 7-day range** for quick analysis
2. **Check GEX first** for risk assessment
3. **Monitor flip point daily** for key levels
4. **Compare C/P ratio** with price action
5. **Watch skew changes** for regime shifts
6. **Export data regularly** for historical records
7. **Use cache** for repeated queries
8. **Analyze multiple timeframes** (7d, 30d, 90d)

---

## 🔑 Keyboard Shortcuts

- **R**: Rerun current page
- **C**: Clear cache
- **S**: Show/hide sidebar
- **Ctrl+C**: Stop Streamlit server

---

## 📈 Analysis Checklist

### Pre-Trade Analysis
- [ ] Check squeeze pressure score
- [ ] Identify gamma flip point
- [ ] Review C/P ratio sentiment
- [ ] Analyze IV skew for pricing
- [ ] Calculate expected Greeks for position

### Risk Management Check
- [ ] Review portfolio Greeks
- [ ] Monitor gamma for rehedging
- [ ] Assess vega exposure
- [ ] Track theta decay
- [ ] Check delta neutrality

### Daily Routine
- [ ] Fetch latest data (7-day range)
- [ ] Check gamma squeeze risk
- [ ] Review volume and flow
- [ ] Monitor volatility changes
- [ ] Update portfolio Greeks
- [ ] Export data for records

---

## 📞 Quick Links

- **GitHub**: https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit
- **Issues**: https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit/issues
- **Deribit**: https://www.deribit.com
- **Full Guide**: [USAGE_GUIDE.md](USAGE_GUIDE.md)

---

## 🎯 Remember

1. **Always run analysis first** before viewing pages
2. **GEX is critical** for understanding market dynamics
3. **Greeks change daily** - monitor regularly
4. **Cache speeds up** repeated queries
5. **Export data** for external analysis

---

**Last Updated**: December 11, 2025
**Version**: 1.0.0
**Dashboard**: BTC Options Analysis
