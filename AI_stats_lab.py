import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    x = np.asarray(x)
    pdf = lam * np.exp(-lam * x)
    return np.where(x >= 0, pdf, 0.0)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    lower = max(a, 0)
    upper = max(b, 0)
    if upper <= lower:
        return 0.0
    return np.exp(-lam * lower) - np.exp(-lam * upper)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1.0, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    x = np.asarray(x)
    coef = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    exp_term = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coef * exp_term


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """
    # Use unnormalized likelihoods since constants cancel in the ratio.
    like_a = np.exp(-((time - 40) ** 2) / 4)
    like_b = np.exp(-((time - 45) ** 2) / 4)
    num = 0.7 * like_b
    den = 0.3 * like_a + num
    return num / den


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    # Approximate with a small window around the time value.
    eps = 0.1
    labels = np.random.choice(["A", "B"], size=n, p=[0.3, 0.7])
    samples = np.empty(n)
    sigma = np.sqrt(2)
    mask_a = labels == "A"
    mask_b = ~mask_a
    samples[mask_a] = np.random.normal(loc=40, scale=sigma, size=mask_a.sum())
    samples[mask_b] = np.random.normal(loc=45, scale=sigma, size=mask_b.sum())

    in_window = (samples >= time - eps) & (samples <= time + eps)
    if not np.any(in_window):
        return 0.0
    return np.mean(labels[in_window] == "B")
