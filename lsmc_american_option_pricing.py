import numpy as np
import scipy.stats as stats
from datetime import datetime


class AmericanOptionsLSMC:
    """
    Class for American options pricing using Longstaff-Schwartz (2001):
    "Valuing American Options by Simulation: A Simple Least-Squares Approach."
    """

    def __init__(self, S0, strike, T, M, r, div, sigma, option_type='call', simulations=10000):
        self.S0 = S0  # Initial stock price
        self.strike = strike  # Strike price
        self.T = T  # Time to maturity
        self.M = M  # Number of time steps
        self.r = r  # Risk-free rate
        self.div = div  # Dividend yield
        self.sigma = sigma  # Volatility
        self.option_type = option_type.lower()  # 'call' or 'put'
        self.dt = T / M  # Time step size
        self.discount = np.exp(-r * self.dt)
        self.simulations = simulations

    def simulate_stock_paths(self, N):
        np.random.seed(42)
        dt = self.T / self.M
        stock_paths = np.zeros((N, self.M + 1))
        stock_paths[:, 0] = self.S0
        for t in range(1, self.M + 1):
            z = np.random.standard_normal(N)
            stock_paths[:, t] = stock_paths[:, t - 1] * np.exp(
                (self.r - self.div - 0.5 * self.sigma ** 2) *
                dt + self.sigma * np.sqrt(dt) * z
            )
        return stock_paths

    @property
    def price(self):
        stock_paths = self.simulate_stock_paths(self.simulations)
        payoffs = np.maximum(
            (self.strike - stock_paths) if self.option_type == 'put' else (stock_paths - self.strike),
            0
        )
        cashflows = payoffs[:, -1]
        for t in range(self.M - 1, 0, -1):
            in_the_money = payoffs[:, t] > 0
            regression = np.polyfit(
                stock_paths[in_the_money, t],
                cashflows[in_the_money] * self.discount,
                2
            )
            continuation_values = np.polyval(
                regression, stock_paths[in_the_money, t])
            exercise_values = payoffs[in_the_money, t]
            cashflows[in_the_money] = np.where(
                exercise_values > continuation_values, exercise_values, cashflows[in_the_money] * self.discount)
        option_price = np.mean(cashflows) * self.discount
        return option_price

    @property
    def delta(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.S0 + diff, self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma, self.option_type, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0 - diff, self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma, self.option_type, self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def gamma(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.S0 + diff, self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma, self.option_type, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0 - diff, self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma, self.option_type, self.simulations)
        return (myCall_1.delta - myCall_2.delta) / float(2. * diff)

    @property
    def vega(self):
        diff = self.sigma * 0.01
        myCall_1 = AmericanOptionsLSMC(self.S0, self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma + diff, self.option_type, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0, self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma - diff, self.option_type, self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def rho(self):
        diff = self.r * 0.01
        myCall_1 = AmericanOptionsLSMC(self.S0, self.strike, self.T, self.M,
                                       self.r + diff, self.div, self.sigma, self.option_type, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0, self.strike, self.T, self.M,
                                       self.r - diff, self.div, self.sigma, self.option_type, self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def theta(self):
        diff = 1 / 252.
        myCall_1 = AmericanOptionsLSMC(self.S0, self.strike, self.T + diff, self.M,
                                       self.r, self.div, self.sigma, self.option_type, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0, self.strike, self.T - diff, self.M,
                                       self.r, self.div, self.sigma, self.option_type, self.simulations)
        return (myCall_2.price - myCall_1.price) / float(2. * diff)


# Example Usage
if __name__ == '__main__':
    S0 = 243.54  # Current stock price from AAPL
    strike = 230  # Strike price
    today = datetime(2025, 1, 3)
    expiration = datetime(2025, 1, 10)
    T = (expiration - today).days / 365  # Time to maturity in years
    M = 50  # Number of time steps
    r = 0.05  # Risk-free rate
    div = 0.0  # Dividend yield (assumed 0 for now)
    sigma = 0.4229  # Implied Volatility (42.29%)
    N = 10000  # Number of Monte Carlo simulations

    option_pricing = AmericanOptionsLSMC(
        S0, strike, T, M, r, div, sigma, option_type='call')
    price = option_pricing.price
    print(f"LSMC Option Price: {price:.2f}")
    print(f"Delta: {option_pricing.delta:.4f}")
    print(f"Gamma: {option_pricing.gamma:.4f}")
    print(f"Theta: {option_pricing.theta:.4f}")
    print(f"Vega: {option_pricing.vega:.4f}")
    print(f"Rho: {option_pricing.rho:.4f}")
