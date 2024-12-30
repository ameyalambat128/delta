import numpy as np


class AmericanOptionBinomialTree:
    def __init__(self, S, K, T, r, sigma, n, option_type='call'):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity (in years)
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.n = n  # Number of time steps
        self.option_type = option_type.lower()  # 'call' or 'put'
        self.dt = T / n  # Time step size
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Up movement factor
        self.d = 1 / self.u  # Down movement factor
        self.p = (np.exp(r * self.dt) - self.d) / \
            (self.u - self.d)  # Risk-neutral probability
        self.one_minus_p = 1 - self.p

    def price(self, american=True):
        # Step 1: Initialize stock price tree
        stock_price = self.S * \
            self.u ** np.arange(self.n + 1) * \
            self.d ** np.arange(self.n, -1, -1)

        # Step 2: Initialize option values at maturity
        if self.option_type == 'call':
            option_values = np.maximum(stock_price - self.K, 0)
        else:
            option_values = np.maximum(self.K - stock_price, 0)

        # Step 3: Backward Induction
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


# Example Usage
if __name__ == '__main__':
    option = AmericanOptionBinomialTree(
        S=100, K=100, T=1, r=0.05, sigma=0.2, n=200, option_type='call')
    price = option.price()
    print(f"The price of the American Call Option is: {price:.2f}")
