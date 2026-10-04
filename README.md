# Cuarta lista de ejercicios — Introducción al Machine Learning

**Curso:** Introducción al Machine Learning — Universidad El Bosque (2026-2)
**Integrante:** Daniel Vargas

## Estructura del repositorio

```
lista_4/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- data/                  datos crudos usados por los 4 ejercicios
|-- src/lista4/            paquete local con el codigo reutilizable
|   |-- data.py            carga de CSV y extraccion de (X, y)
|   |-- models.py          ajuste de regresion lineal y logistica
|   '-- metrics.py         riesgo cuadratico, riesgo logistico, errores de clasificacion
'-- notebooks/              un cuaderno ejecutado por ejercicio
    |-- ejercicio_1.ipynb
    |-- ejercicio_2.ipynb
    |-- ejercicio_3.ipynb
    '-- ejercicio_4.ipynb
```

Todas las operaciones que se repiten entre ejercicios (cargar un CSV, ajustar
un modelo lineal o logístico, calcular el riesgo empírico, clasificar y
contar errores) están implementadas una sola vez en `src/lista4/` y se
importan desde cada cuaderno; ningún cuaderno redefine estas funciones.

## Instalación

Con Python 3.14+ instalado, desde la raíz de `lista_4/`:

```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
pip install -e .
```

El último comando instala el paquete local `lista4` (definido en
`pyproject.toml`, código real en `src/lista4/`) en modo editable, de forma
que los cuadernos puedan hacer `from lista4.data import ...` sin trucos de
rutas y reflejando inmediatamente cualquier cambio en `src/lista4/`.

## Ejecución

Para reproducir un cuaderno desde un kernel reiniciado:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/ejercicio_1.ipynb
```

(cambiando el nombre del archivo para los ejercicios 2, 3 y 4), o de forma
interactiva con `jupyter notebook notebooks/` y "Kernel → Restart & Run All".

## Contenido de cada ejercicio

- **Ejercicio 1** (`experimento_1.csv`): regresión lineal simple vs. múltiple,
  correlaciones, gráficas con la recta ajustada y selección de la mejor
  variable individual.
- **Ejercicio 2** (`experimento_2_a/b/c.csv`): efecto de agregar `x2` bajo
  distintos niveles de colinealidad con `x1`.
- **Ejercicio 3** (`experimento_3.csv`): invariancia de la regresión lineal
  ante el reescalamiento de una variable.
- **Ejercicio 4** (`clientes.csv`): regresión logística para predecir
  abandono, comparando un modelo con todas las variables contra uno que solo
  usa variables disponibles en el momento de la predicción (sin fuga de
  información del futuro).
