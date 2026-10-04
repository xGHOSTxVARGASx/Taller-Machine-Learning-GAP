"""
models.py
---------
Funciones reutilizables para ajustar los modelos usados en los 4 ejercicios:
regresión lineal (Ejercicios 1, 2 y 3) y regresión logística (Ejercicio 4).

Se usa scikit-learn, pero toda la interacción con scikit-learn queda
encapsulada aquí: los cuadernos nunca llaman directamente a
LinearRegression o LogisticRegression, sino a estas funciones.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def ajustar_regresion_lineal(X, y):
    """
    Ajusta un modelo de regresión lineal con intercepto.

    Parámetros
    ----------
    X : array_like de forma (n_muestras, n_variables)
    y : array_like de forma (n_muestras,)

    Retorna
    -------
    modelo : LinearRegression ya ajustado (modelo.intercept_, modelo.coef_)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo


def ajustar_regresion_logistica(X, y, C=1e6, max_iter=10000):
    """
    Ajusta un modelo de regresión logística con solver='lbfgs', como pide
    el enunciado del Ejercicio 4.

    Parámetros
    ----------
    X : array_like de forma (n_muestras, n_variables)
    y : array_like de forma (n_muestras,), valores en {0, 1}
    C : float
        Inverso de la regularización; C=1e6 equivale a (casi) no regularizar,
        como pide el enunciado.
    max_iter : int
        Número máximo de iteraciones del optimizador.

    Retorna
    -------
    modelo : LogisticRegression ya ajustado
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    modelo = LogisticRegression(solver="lbfgs", C=C, max_iter=max_iter)
    modelo.fit(X, y)
    return modelo
