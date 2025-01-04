import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


class YFinanceOptionsData:
    def __init__(self, ticker, expiration_date):
        self.ticker = ticker
        self.expiration_date = expiration_date
        self.stock = yf.Ticker(ticker)
        self.stock_data = self.stock.history(period='1d')

        if self.stock_data.empty:
            raise ValueError(
                "Failed to retrieve stock data. Check the ticker symbol.")

        self.current_price = self.stock_data['Close'].iloc[-1]
        self.risk_free_rate = 0.05  # Default risk-free rate

        today = datetime.now()
        expiration = datetime.strptime(expiration_date, '%Y-%m-%d')
        self.time_to_maturity = (expiration - today).days / 365

        if self.time_to_maturity <= 0:
            raise ValueError("Expiration date must be in the future.")

        self.options_chain = self.stock.option_chain(expiration_date)
        self.calls = self.options_chain.calls
        self.puts = self.options_chain.puts

        self.calls_data = self._extract_option_data(self.calls)
        self.puts_data = self._extract_option_data(self.puts)

    def _extract_option_data(self, options_df):
        data = []
        for _, row in options_df.iterrows():
            data.append({
                'Strike': row['strike'],
                'Last Price': row['lastPrice'],
                'Net Change': row['change'],
                'Bid': row['bid'],
                'Ask': row['ask'],
                'Volume': row['volume'],
                'Open Interest': row['openInterest'],
                'IV': row['impliedVolatility'],
                'In The Money': row['inTheMoney'],
                'Last Trade Date': row['lastTradeDate']
            })
        return pd.DataFrame(data)

    def get_calls_data(self):
        return self.calls_data

    def get_puts_data(self):
        return self.puts_data

    def display_calls(self):
        print("\nCall Options Data:")
        print(self.calls_data.to_string(index=False))

    def display_puts(self):
        print("\nPut Options Data:")
        print(self.puts_data.to_string(index=False))


# Example Usage
if __name__ == '__main__':
    ticker = 'AAPL'
    expiration_date = '2025-01-31'
    options_data = YFinanceOptionsData(ticker, expiration_date)
    options_data.display_calls()
    options_data.display_puts()
