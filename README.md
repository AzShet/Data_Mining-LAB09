# Data Mining - LAB09 📊🧠

Este proyecto contiene la implementación y prueba unitaria de funciones de preprocesamiento y modelos de clasificación aplicados a conjuntos de datos clásicos como MNIST y datos binarios. Además, se ha configurado una arquitectura de proyecto profesional compatible con `pytest` y entornos de Jupyter Notebook para asegurar la reproducibilidad y mantenibilidad del código.

---

## 📁 Estructura del Proyecto

```plaintext
Data_Mining-LAB09/
├── pytest.ini                      # Configuración global para pytest
├── src/
│   ├── __init__.py                # Hace a 'src' un paquete de Python
│   ├── utils.py                   # Contiene funciones de carga, procesamiento y modelos
│   └── LAB09-RUELAS.ipynb         # Notebook principal para desarrollo y pruebas visibles
├── tests/
│   └── test_utils.py              # Pruebas unitarias con pytest
├── .pytest_cache/                 # Caché generada por pytest
└── README.md                      # Este archivo
````

---

## ⚙️ Requisitos del Entorno

* Python 3.12+
* Paquetes requeridos:

```bash
pip install numpy scikit-learn tensorflow keras pytest
```

---

## 🔬 Descripción de Funcionalidades

### `utils.py`

Contiene las siguientes funciones:

* `cargar_datos_clasificacion()`: carga datos binarios y retorna `X` (features) y `y` (labels) como `np.ndarray`.
* `crear_modelo_clasificacion()`: instancia un `MLPClassifier` de `sklearn` para clasificación binaria.
* `preprocesar_mnist()`: descarga MNIST desde `keras.datasets`, normaliza y convierte las etiquetas a codificación one-hot.
* `crear_cnn()`: define una arquitectura CNN simple usando `keras.models.Sequential`.

---

## 🧪 Pruebas Unitarias con Pytest

### Ubicación

Todas las pruebas se encuentran en:

```
tests/test_utils.py
```

### Cobertura

Se testean las funciones principales de `src.utils`:

* **Datos de Clasificación**

  * Verifica tipo y forma de `X`, `y`.
  * Valida que `y` contenga solo `0` y `1`.

* **Modelo MLP**

  * Asegura que sea instancia de `MLPClassifier`.
  * Confirma presencia de métodos `.fit()` y `.predict()`.

* **Preprocesamiento MNIST**

  * Verifica forma `(28, 28, 1)` y tipo `float32`.
  * Asegura que el máximo sea `≤ 1.0` y etiquetas estén one-hot.

* **Modelo CNN**

  * Asegura que tenga `.fit()` y `.evaluate()`.
  * Verifica `input_shape == (None, 28, 28, 1)`.

### Ejecución desde Notebook (`LAB09-RUELAS.ipynb`)

Debido a que los notebooks corren en un subdirectorio (`src/`), es necesario configurar correctamente el entorno de ejecución.

#### 📍 Paso 1: Establecer el directorio raíz del proyecto y `PYTHONPATH`

```python
import os
import sys

# Ruta raíz del proyecto
project_root_path = r'c:\Users\AzShet\Documents\Jupyter_LAB\jupyter_projects\5to_ciclo\DataMining\lab9\Data_Mining-LAB09'

# Cambiar el directorio de trabajo
%cd {project_root_path}
print(f"Directorio de trabajo actual establecido en: {os.getcwd()}")

# Configurar PYTHONPATH para incluir el directorio raíz
os.environ['PYTHONPATH'] = project_root_path + os.pathsep + os.environ.get('PYTHONPATH', '')
print(f"PYTHONPATH establecido en: {os.environ['PYTHONPATH']}")
```

#### 📍 Paso 2: Ejecutar pytest desde el Notebook

```python
!pytest tests/test_utils.py -v
```

Esto permite que las pruebas se ejecuten correctamente desde el notebook y sus resultados se visualicen en forma de salida verde/roja, tal como se solicita en prácticas académicas.

---

## 🧾 Contenido del Archivo `pytest.ini`

```ini
# pytest.ini
[pytest]
pythonpath = src
```

Este archivo configura `pytest` para reconocer el paquete `src` al momento de importar, útil cuando las pruebas se corren desde la raíz del proyecto.

---

## 🧼 Limpieza de Caché y Precauciones

En caso de errores de importación persistentes, se recomienda eliminar los directorios `__pycache__/` tanto en `src/` como en `tests/`, para forzar una recompilación de módulos.

```bash
find . -type d -name "__pycache__" -exec rm -r {} +
```

En Windows:

```powershell
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

---

## ✅ Buenas Prácticas Observadas

* Separación de lógica (`src/`) y pruebas (`tests/`).
* Uso de `pytest.ini` para configuración portable.
* Notebook configurado para ejecutar pruebas unitarias con salidas visibles.
* Documentación del entorno y configuración de importación vía `PYTHONPATH`.

---

## 📚 Créditos

**Autor:** [César Diego Ruelas Flores](https://www.linkedin.com/in/diego-ruelas-flores/)
**Profesor**: [Luis Paraguay Arzapalo](https://github.com/luispar90) – Curso *Minería de Datos*.
**Curso:** Big Data y Ciencia de Datos – TECSUP
**Laboratorio:** LAB09 – Minería de Datos
**Fecha creación:** 15 de mayo de 2025
**Fecha de ultima actualización:** 4 de junio de 2025