# Gradient Descent

Notebooks that build gradient descent from scratch:

- `GradientDescent.ipynb` – simple and multivariate linear regression (batch GD)
- `GD-MiniBatch-Stochastic.ipynb` – mini-batch and stochastic GD
- `GD-Momentum-NAG.ipynb` – momentum and Nesterov accelerated gradient
- `Adagrad-RMSProp-Adam.ipynb` – adaptive optimisers

`gradient_descent.py` holds the reusable, vectorised implementation (numpy only).
Run `python gradient_descent.py` to check it against the closed-form least-squares solution.
Data: `RegData.csv`, `MultipleLR.csv`.
