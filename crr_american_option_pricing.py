import numpy as np
from datetime import datetime


class CRRBinomialOptionPricing:
    def __init__(self, S, K, T, r, sigma, n, option_type='call', american=True):
        self.S = S  # Underlying stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility (from yfinance)
        self.n = n  # Number of time steps
        self.option_type = option_type.lower()  # 'call' or 'put'
        self.american = american  # American or European option boolean

        # Time step
        self.dt = self.T / self.n
        # Up and Down movement factors
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        # Risk-neutral probability
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.one_minus_p = 1 - self.p

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, value):
        self._S = value

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value

    @property
    def option_type(self):
        return self._option_type

    @option_type.setter
    def option_type(self, value):
        self._option_type = value.lower()

    @property
    def american(self):
        return self._american

    @american.setter
    def american(self, value):
        self._american = value

    def build_stock_price_tree(self):
        stock_price = self.S * \
            self.u ** np.arange(self.n + 1) * \
            self.d ** np.arange(self.n, -1, -1)
        return stock_price

    def option_payoff(self, stock_price):
        if self.option_type == 'call':
            return np.maximum(stock_price - self.K, 0)
        else:
            return np.maximum(self.K - stock_price, 0)

    def price(self):
        stock_price = self.build_stock_price_tree()
        option_values = self.option_payoff(stock_price)

        for i in range(self.n - 1, -1, -1):
            stock_price = stock_price[:-1] * self.u
            option_values = np.exp(-self.r * self.dt) * (
                self.p * option_values[1:] +
                self.one_minus_p * option_values[:-1]
            )
            if self.american:
                if self.option_type == 'call':
                    option_values = np.maximum(
                        option_values, stock_price - self.K)
                else:
                    option_values = np.maximum(
                        option_values, self.K - stock_price)

        return option_values[0]

    def greeks(self):
        epsilon = 0.01

        # Delta
        price_up = CRRBinomialOptionPricing(
            self.S * (1 + epsilon), self.K, self.T, self.r, self.sigma, self.n, self.option_type, self.american).price()
        price_down = CRRBinomialOptionPricing(
            self.S * (1 - epsilon), self.K, self.T, self.r, self.sigma, self.n, self.option_type, self.american).price()
        delta = (price_up - price_down) / (2 * self.S * epsilon)

        # Gamma
        price_up2 = CRRBinomialOptionPricing(
            self.S * (1 + 2 * epsilon), self.K, self.T, self.r, self.sigma, self.n, self.option_type, self.american).price()
        price_down2 = CRRBinomialOptionPricing(
            self.S * (1 - 2 * epsilon), self.K, self.T, self.r, self.sigma, self.n, self.option_type, self.american).price()
        gamma = (price_up2 - 2 * self.price() + price_down2) / \
            (self.S * epsilon) ** 2

        # Theta
        theta = (CRRBinomialOptionPricing(self.S, self.K, self.T - self.dt, self.r, self.sigma,
                 self.n, self.option_type, self.american).price() - self.price()) / self.dt

        # Vega
        price_vega = CRRBinomialOptionPricing(
            self.S, self.K, self.T, self.r, self.sigma + epsilon, self.n, self.option_type, self.american).price()
        vega = (price_vega - self.price()) / epsilon

        # Rho
        price_rho = CRRBinomialOptionPricing(
            self.S, self.K, self.T, self.r + epsilon, self.sigma, self.n, self.option_type, self.american).price()
        rho = (price_rho - self.price()) / epsilon

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }


# Example Usage
if __name__ == '__main__':
    # Example using hardcoded values from yfinance
    S = 243.54  # Current stock price from yfinance
    K = 230  # Strike price from yfinance
    today = datetime(2025, 1, 6)
    expiration = datetime(2025, 3, 21)
    # Time to maturity in years from yfinance
    T = (expiration - today).days / 365
    r = 0.05  # Risk-free rate from yfinance
    sigma = 0.2495  # IV from yfinance
    n = 200  # Number of time steps

    option_pricing = CRRBinomialOptionPricing(
        S=S, K=K, T=T, r=r, sigma=sigma, n=n, option_type='call', american=True)
    price = option_pricing.price()
    greeks = option_pricing.greeks()
    print(f"Option Price: {price:.2f}")
    print("Greeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.4f}")
