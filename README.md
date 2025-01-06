# Delta

Delta is a cutting-edge **Options Pricing and Analytics Platform** designed to provide accurate, real-time options pricing and comprehensive analytics. Built on robust financial models like **Cox-Ross-Rubinstein (CRR)** and **Least-Squares Monte Carlo (LSMC)**, Delta aims to empower traders, analysts, and financial enthusiasts with precise insights into the world of options trading.

## **Core Features**

### **Options Pricing Models**

- **Cox-Ross-Rubinstein (CRR)**: Accurate pricing using binomial trees.
- **Least-Squares Monte Carlo (LSMC)**: Regression-based option pricing for American options.

### **Greeks Calculation**

- Real-time computation of **Delta**, **Gamma**, **Vega**, **Theta**, and **Rho**.
- Sensitivity analysis using advanced finite difference methods.

### **Market Data Integration**

- Real-time stock and options data from Yahoo Finance.
- Support for live option chains with detailed analytics.

### **Technology Stack**

- **Python**: Core backend logic.
- **uv**: An extremely fast Python package and project manager, written in Rust.
- **NumPy** & **SciPy**: Numerical computations.
- **Pandas**: Data manipulation and visualization.
- **yfinance**: Real-time market data.
- **Matplotlib**: Visual representation of options data.

## **How to Use**

### Installation

```bash
# Clone the repository
git clone https://github.com/ameyalambat128/delta.git

# Navigate to the project folder
cd delta

# Install dependencies (uv)
uv sync
```

or if you're not using `uv`

```bash
# Create a virtual env
python -m venv .venv

# Activate virtual env
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
