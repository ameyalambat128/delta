import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


class MarketData:
    def __init__(self, ticker, expiration_date):
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period='1d')

        if stock_data.empty:
            raise ValueError(
                "Failed to retrieve stock data. Check the ticker symbol.")

        self.S = stock_data['Close'].iloc[-1]  # Current stock price

        today = datetime.now()
        expiration = datetime.strptime(expiration_date, '%Y-%m-%d')
        delta = expiration - today
        self.T = delta.days / 365  # Time to maturity (in years)

        if self.T <= 0:
            raise ValueError("Expiration date must be in the future.")

        options_chain = stock.option_chain(expiration_date)
        self.calls = options_chain.calls
        self.puts = options_chain.puts

        self.strikes = self.calls['strike'].values
        self.volumes = self.calls['volume'].values
        self.IV = self.calls['impliedVolatility'].values
        self.last_prices = self.calls['lastPrice'].values
        self.bid_prices = self.calls['bid'].values
        self.ask_prices = self.calls['ask'].values
        self.open_interest = self.calls['openInterest'].values
        self.in_the_money = self.calls['inTheMoney'].values
        self.last_trade_date = self.calls['lastTradeDate'].values
        self.net_change = self.calls['change'].values

        # Default to first available strike
        self.K = self.strikes[0] if len(self.strikes) > 0 else None
        if self.K is None:
            raise ValueError("No valid strike prices available.")

        self.r = 0.05  # Default risk-free rate
        self.sigma = self.IV[0] if len(
            self.IV) > 0 and not np.isnan(self.IV[0]) else 0.2


class OptionData:
    def __init__(self, market_data: MarketData, option_type='call'):
        self.market_data = market_data
        self.option_type = option_type.lower()
        self.strikes = market_data.strikes
        self.volumes = market_data.volumes
        self.implied_volatility = market_data.IV
        self.last_prices = market_data.last_prices
        self.bid_prices = market_data.bid_prices
        self.ask_prices = market_data.ask_prices
        self.open_interest = market_data.open_interest
        self.in_the_money = market_data.in_the_money
        self.last_trade_date = market_data.last_trade_date
        self.net_change = market_data.net_change
        self.r = market_data.r
        self.sigma = market_data.sigma
        self.S = market_data.S
        self.T = market_data.T
        self.K = market_data.K


class AmericanOptionBinomialTree:
    def __init__(self, option_data: OptionData, n=200):
        self.S = option_data.S
        self.K = option_data.K
        self.T = option_data.T
        self.r = option_data.r
        self.sigma = option_data.sigma
        self.n = n
        self.option_type = option_data.option_type
        self.dt = self.T / n
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.one_minus_p = 1 - self.p

    def price(self, american=True):
        stock_price = self.S * \
            self.u ** np.arange(self.n + 1) * \
            self.d ** np.arange(self.n, -1, -1)

        if self.option_type == 'call':
            option_values = np.maximum(stock_price - self.K, 0)
        else:
            option_values = np.maximum(self.K - stock_price, 0)

        for i in range(self.n - 1, -1, -1):
            stock_price = stock_price[:-1] * self.u
            option_values = np.exp(-self.r * self.dt) * (
                self.p * option_values[1:] +
                self.one_minus_p * option_values[:-1]
            )

            if american:
                if self.option_type == 'call':
                    option_values = np.maximum(
                        option_values, stock_price - self.K)
                else:
                    option_values = np.maximum(
                        option_values, self.K - stock_price)

        return option_values[0]

    def greeks(self):
        delta = (self.price() - self.price()) / (self.S * 0.01)
        gamma = (delta - delta) / (self.S * 0.01)
        theta = -(self.price() - self.price()) / (self.T / self.n)
        vega = (self.price() - self.price()) / (self.sigma * 0.01)
        rho = (self.price() - self.price()) / (self.r * 0.01)

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }


class OptionsChainDashboard:
    def __init__(self, market_data: MarketData, option_type='call', n=200):
        self.market_data = market_data
        self.option_type = option_type.lower()
        self.n = n
        self.data = []

        for i in range(len(market_data.strikes)):
            option_data = OptionData(
                market_data=market_data, option_type=option_type)
            option_data.K = market_data.strikes[i]
            option_data.sigma = market_data.IV[i] if not np.isnan(
                market_data.IV[i]) else 0.2

            model = AmericanOptionBinomialTree(option_data, n)
            price = model.price()
            greeks = model.greeks()

            self.data.append({
                'Strike': option_data.K,
                'Last Price': market_data.last_prices[i],
                'Net Change': market_data.net_change[i],
                'Bid': market_data.bid_prices[i],
                'Ask': market_data.ask_prices[i],
                'Volume': market_data.volumes[i],
                'Open Interest': market_data.open_interest[i],
                'IV': market_data.IV[i],
                'Model Price': price,
                **greeks
            })

    def display(self):
        df = pd.DataFrame(self.data)
        print(df)


if __name__ == '__main__':
    market_data = MarketData('AAPL', '2025-01-31')
    options_chain = OptionsChainDashboard(market_data, 'call')
    options_chain.display()
