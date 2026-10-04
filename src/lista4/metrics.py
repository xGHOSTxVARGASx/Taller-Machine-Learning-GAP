"""
metrics.py
----------
Funciones reutilizables de riesgo empírico y métricas de clasificación,
compartidas entre los 4 cuadernos.
"""

import numpy as np


def riesgo_cuadratico(y_true, y_pred):
    """
    Riesgo empírico con pérdida cuadrática:

        R_S(h) = (1/n) * sum_i (y_i - h(x_i))^2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def riesgo_logistico(y_true, p_hat, eps=1e-12):
    """
    Riesgo empírico logístico (log-loss promedio):

        R_S(h) = -(1/n) * sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ]

    Se recorta p_hat a [eps, 1-eps] para evitar log(0) por redondeo numérico
    cuando el modelo predice probabilidades extremadamente cercanas a 0 o 1.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_hat = np.clip(np.asarray(p_hat, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p_hat) + (1.0 - y_true) * np.log(1.0 - p_hat)))


def clasificar(p_hat, umbral=0.5):
    """
    Convierte probabilidades estimadas en clases {0, 1} usando el umbral
    p_hat >= umbral, como pide el enunciado del Ejercicio 4.
    """
    return (np.asarray(p_hat, dtype=float) >= umbral).astype(int)


def errores_clasificacion(y_true, y_pred_clase):
    """
    Cuenta los errores de clasificación y su proporción.

    Retorna
    -------
    n_errores : int
    proporcion_errores : float
    """
    y_true = np.asarray(y_true)
    y_pred_clase = np.asarray(y_pred_clase)
    n_errores = int(np.sum(y_true != y_pred_clase))
    proporcion_errores = n_errores / len(y_true)
    return n_errores, proporcion_errores


def rango(valores):
    """
    rango(z) = max(z) - min(z), como se define en el Ejercicio 2(e).
    """
    valores = np.asarray(valores, dtype=float)
    return float(np.max(valores) - np.min(valores))
