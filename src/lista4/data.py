"""
data.py
-------
Funciones reutilizables para cargar los datos de data/ desde los cuadernos,
sin repetir rutas ni lógica de lectura en cada notebook.
"""

from pathlib import Path
import pandas as pd

# data/ está dos niveles arriba de este archivo: lista_4/src/lista4/data.py
# -> parents[0] = lista4, parents[1] = src, parents[2] = lista_4 (raíz del repo)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def cargar_csv(nombre_archivo, data_dir=DATA_DIR):
    """
    Carga un archivo CSV desde la carpeta data/ (o desde data_dir si se
    especifica otra ruta) y lo retorna como un DataFrame de pandas.
    """
    ruta = Path(data_dir) / nombre_archivo
    return pd.read_csv(ruta)


def obtener_xy(df, columnas_x, columna_y):
    """
    Extrae de un DataFrame las columnas predictoras (columnas_x, lista de
    nombres) y la columna objetivo (columna_y, nombre), como arreglos de
    numpy listos para pasar a los modelos de src/lista4/models.py.

    X siempre se retorna con forma (n_muestras, n_variables), incluso si
    columnas_x tiene un solo elemento.
    """
    if isinstance(columnas_x, str):
        columnas_x = [columnas_x]
    X = df[columnas_x].to_numpy(dtype=float)
    y = df[columna_y].to_numpy(dtype=float)
    return X, y
