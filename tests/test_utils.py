import numpy as np
from sklearn.neural_network import MLPClassifier
from src.utils import cargar_datos_clasificacion, crear_modelo_clasificacion, preprocesar_mnist, crear_cnn
import pytest


def test_cargar_datos_clasificacion():
    X, y = cargar_datos_clasificacion()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0
    assert set(np.unique(y)).issubset({0, 1})


def test_crear_modelo_clasificacion():
    modelo = crear_modelo_clasificacion()
    assert isinstance(modelo, MLPClassifier)
    assert hasattr(modelo, 'fit')
    assert hasattr(modelo, 'predict')


def test_preprocesar_mnist():
    X_train, X_test, y_train, y_test = preprocesar_mnist()
    assert X_train.shape[1:] == (28, 28, 1)
    assert X_train.dtype == np.float32
    assert X_train.max() <= 1.0
    assert y_train.shape[1] == 10  # One-hot


def test_crear_cnn():
    cnn = crear_cnn()
    assert hasattr(cnn, 'fit')
    assert hasattr(cnn, 'evaluate')
    assert cnn.input_shape == (None, 28, 28, 1)
