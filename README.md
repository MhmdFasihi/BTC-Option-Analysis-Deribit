# 📊 BTC Options Analysis - Deribit

Comprehensive cryptocurrency options analytics for Bitcoin (BTC) and Ethereum (ETH) using Deribit exchange data.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- ✅ **Real-time Data Fetching**: Parallel data collection from Deribit API with intelligent caching
- ✅ **Greeks Analysis**: Complete calculation of Delta, Gamma, Vega, and Theta
- ✅ **Volatility Analytics**:
  - 3D Implied Volatility surfaces
  - IV skew analysis by maturity
  - Historical vs Implied volatility comparison
  - Weighted IV time series
- ✅ **Gamma Exposure (GEX)**: Net gamma exposure profiles by strike price
- ✅ **Interactive Visualizations**: 10+ charts built with Plotly
- ✅ **Multi-Currency Support**: BTC and ETH analysis
- ✅ **Export Capabilities**: CSV, JSON, and HTML outputs

## 🚀 Quick Start

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit.git
cd BTC-Option-Analysis-Deribit

# Create conda environment
conda env create -f environment.yml
conda activate crypto-option

# Run analysis
python deribit_options_analysis.py
```

### Using Pip

```bash
# Clone the repository
git clone https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit.git
cd BTC-Option-Analysis-Deribit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run analysis
python deribit_options_analysis.py
```

## 📖 Usage

The main script analyzes BTC and ETH options for the last 7 days by default:

```python
from datetime import date, timedelta
from deribit_options_analysis import CryptoOptionsAnalysis

# Set parameters
end_date = date.today()
start_date = end_date - timedelta(days=7)
currencies = ['BTC', 'ETH']

# Run analysis
analysis = CryptoOptionsAnalysis(currencies, start_date, end_date)
analysis.run_analysis()
```

Results are saved to `Option_Analysis_Results/` directory with:
- Volatility surfaces (3D plots)
- Greeks analysis
- Gamma exposure charts
- Distribution analysis
- Summary reports

## 📊 Core Components

### OptionsDataFetcher
- Parallel API data collection with ThreadPoolExecutor
- Intelligent file-based caching
- WebSocket + REST API for historical volatility
- Supports date ranges up to 365 days

### OptionsAnalyzer
- Black-Scholes Greeks calculation (Δ, Γ, ν, Θ)
- Weighted implied volatility analysis
- Volume metrics and Call/Put ratios
- Historical volatility processing

### OptionsVisualizer
- 3D Volatility Surface
- IV Skew by Maturity
- Weighted IV Time Series
- Greeks 3D Surfaces
- Distribution Analysis

### GammaExposureAnalyzer
- Market maker positioning analysis
- Net GEX by strike price
- Call vs Put gamma profiles

## 🔧 Configuration

Edit the `main()` function in `deribit_options_analysis.py`:

```python
def main():
    # Set date range
    end_date = date.today()
    start_date = end_date - timedelta(days=7)  # Change days here

    # Specify currencies
    currencies = ['BTC', 'ETH']  # Add more if supported

    # Run analysis
    analysis = CryptoOptionsAnalysis(currencies, start_date, end_date)
    analysis.run_analysis()
```

## 📁 Output Structure

```
Option_Analysis_Results/
├── BTC/
│   ├── BTC_analysis_report.txt
│   ├── BTC_processed_data.csv
│   ├── volatility_surface.html
│   ├── iv_skew_by_maturity.html
│   ├── gamma_exposure.html
│   └── [more visualizations]
└── ETH/
    └── [same structure as BTC]
```

## 📊 Data Source

All options data is sourced from [Deribit Exchange](https://www.deribit.com) public API:
- Historical trades: `/public/get_last_trades_by_currency_and_time`
- Historical volatility: `/public/get_historical_volatility`

**API Documentation**: [docs.deribit.com](https://docs.deribit.com)

## 🛠️ Development Roadmap

This repository is being enhanced into a **Streamlit dashboard**. See development documentation:

- **Pipeline**: Full development pipeline and architecture
- **Roadmap**: Step-by-step implementation guide
- **GitHub Setup**: Version control workflow

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Deribit** for comprehensive options data API
- **Plotly** for interactive visualizations
- **SciPy** for statistical functions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MhmdFasihi/BTC-Option-Analysis-Deribit/issues)
- **Email**: Contact repository owner

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

**Made with ❤️ for the crypto options community**
