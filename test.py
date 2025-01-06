import numpy as np
import pandas as pd
from options_market_data import YFinanceOptionsData
from crr_american_option_pricing import CRRBinomialOptionPricing


class OptionsPricingDashboard:
    def __init__(self, ticker, expiration_date, option_type='call', n=200):
        self.ticker = ticker
        self.expiration_date = expiration_date
        self.option_type = option_type.lower()
        self.n = n

        # Fetch market data
        self.market_data = YFinanceOptionsData(ticker, expiration_date)
        self.option_data = self.market_data.get_calls_data(
        ) if self.option_type == 'call' else self.market_data.get_puts_data()
        self.results = []

        self._calculate_pricing_and_greeks()

    def _calculate_pricing_and_greeks(self):
        for _, row in self.option_data.iterrows():
            strike = row['Strike']
            iv = row['IV'] if not np.isnan(row['IV']) else 0.2

            option_model = CRRBinomialOptionPricing(
                S=self.market_data.current_price,
                K=strike,
                T=self.market_data.time_to_maturity,
                r=self.market_data.risk_free_rate,
                sigma=iv,
                n=self.n,
                option_type=self.option_type,
                american=True
            )

            price = option_model.price()
            greeks = option_model.greeks()

            self.results.append({
                'Strike': strike,
                'Last Price': row['Last Price'],
                'Net Change': row['Net Change'],
                'Bid': row['Bid'],
                'Ask': row['Ask'],
                'Volume': row['Volume'],
                'Open Interest': row['Open Interest'],
                'IV': iv,
                'Model Price': price,
                'Delta': greeks['Delta'],
                'Gamma': greeks['Gamma'],
                'Theta': greeks['Theta'],
                'Vega': greeks['Vega'],
                'Rho': greeks['Rho']
            })

    def display_dashboard(self):
        df = pd.DataFrame(self.results)
        print(df.to_string(index=False))


# Example Usage
if __name__ == '__main__':
    ticker = 'AAPL'
    expiration_date = '2025-01-31'
    dashboard = OptionsPricingDashboard(
        ticker, expiration_date, option_type='call', n=200)
    dashboard.display_dashboard()
