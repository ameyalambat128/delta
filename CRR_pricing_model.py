import numpy as np


class CRRBinomialOptionPricing:
    def __init__(self, S, K, T, r, sigma, n, option_type='call', american=True):
        self.S = S  # Underlying stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.n = n  # Number of time steps
        self.option_type = option_type.lower()  # 'call' or 'put'
        self.american = american  # American or European option

        # Time step
        self.dt = self.T / self.n
        # Up and Down movement factors
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        # Risk-neutral probability
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.one_minus_p = 1 - self.p

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
        S_up = self.S * self.u
        S_down = self.S * self.d
        price_up = CRRBinomialOptionPricing(
            S_up, self.K, self.T, self.r, self.sigma, self.n, self.option_type, self.american).price()
        price_down = CRRBinomialOptionPricing(
            S_down, self.K, self.T, self.r, self.sigma, self.n, self.option_type, self.american).price()

        delta = (price_up - price_down) / (S_up - S_down)
        gamma = ((price_up - 2 * self.price() + price_down) /
                 ((self.S * (self.u - self.d)) ** 2))
        theta = -(self.price() - price_down) / self.dt
        vega = (CRRBinomialOptionPricing(self.S, self.K, self.T, self.r, self.sigma +
                0.01, self.n, self.option_type, self.american).price() - self.price()) / 0.01
        rho = (CRRBinomialOptionPricing(self.S, self.K, self.T, self.r + 0.01, self.sigma,
               self.n, self.option_type, self.american).price() - self.price()) / 0.01

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }


# Example Usage
if __name__ == '__main__':
    option_pricing = CRRBinomialOptionPricing(
        S=100, K=100, T=1, r=0.05, sigma=0.2, n=100, option_type='call', american=True)
    price = option_pricing.price()
    greeks = option_pricing.greeks()
    print(f"Option Price: {price:.2f}")
    print("Greeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.4f}")
