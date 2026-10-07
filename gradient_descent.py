"""Vectorised batch gradient descent for linear regression (numpy only)."""
import numpy as np


def add_bias(X):
    """Prepend a column of ones so theta[0] acts as the intercept."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize(X):
    """Zero-mean / unit-variance scaling. Returns (X_scaled, mean, std)."""
    X = np.asarray(X, dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def cost_function(Xb, y, theta):
    """J(theta) = 1/(2m) * sum((X @ theta - y)^2)."""
    z = Xb @ theta - y
    return (z @ z) / (2 * len(y))


def gradient_descent(Xb, y, alpha=0.01, n_iters=1000, tol=1e-6):
    """Minimise J by batch gradient descent.

    Xb must already contain the bias column (see add_bias).
    Stops early when the gradient norm falls below ``tol``.
    Returns (theta, cost_history).
    """
    Xb = np.asarray(Xb, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)
    theta = np.zeros(Xb.shape[1])
    history = []
    for _ in range(n_iters):
        z = Xb @ theta - y
        history.append((z @ z) / (2 * m))
        grad = Xb.T @ z / m
        if np.linalg.norm(grad) < tol:
            break
        theta -= alpha * grad  # all parameters updated simultaneously
    return theta, np.array(history)


def r2_score(y, y_pred):
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


if __name__ == "__main__":
    # Self-check against the closed-form least-squares solution.
    reg = np.genfromtxt("RegData.csv", delimiter=",")
    Xb, y = add_bias(reg[:, 0]), reg[:, 1]
    theta, hist = gradient_descent(Xb, y, alpha=0.01, n_iters=200_000, tol=1e-9)
    ref = np.linalg.lstsq(Xb, y, rcond=None)[0]
    print("simple LR  GD:", theta, " closed form:", ref)
    assert np.allclose(theta, ref, atol=1e-4) and np.all(np.diff(hist) <= 1e-12)

    multi = np.genfromtxt("MultipleLR.csv", delimiter=",")
    Xs, mu, sd = standardize(multi[:, :-1])
    Xb, y = add_bias(Xs), multi[:, -1]
    theta, hist = gradient_descent(Xb, y, alpha=0.1, n_iters=10_000, tol=1e-9)
    ref = np.linalg.lstsq(Xb, y, rcond=None)[0]
    print("multi LR   GD:", theta, " closed form:", ref)
    print("R2:", r2_score(y, Xb @ theta))
    assert np.allclose(theta, ref, atol=1e-4)
    print("OK")
