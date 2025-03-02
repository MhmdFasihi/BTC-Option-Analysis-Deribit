# Cryptocurrency Options Analysis

## Overview

This repository contains a Python implementation for analyzing cryptocurrency options data, specifically for Bitcoin (BTC) and Ethereum (ETH). The code fetches options data from the Deribit exchange, processes it, calculates various financial metrics (Greeks), and generates visualizations and reports. The results are saved in a directory named `Option_Analysis_Results`, which is created in the current working directory.

## Features

- Fetches options data from the Deribit API.
- Validates input parameters for supported currencies and date ranges.
- Calculates Greeks (Delta, Gamma, Vega, Theta) for options.
- Generates detailed analysis reports and visualizations.
- Saves results in a structured directory format.

## Requirements

To run this code, you will need:

- Python 3.7 or higher
- Required Python packages:
  - `numpy`
  - `pandas`
  - `requests`
  - `plotly`
  - `scipy`
  - `tqdm`
  - `json`
  
You can install the required packages using pip:
