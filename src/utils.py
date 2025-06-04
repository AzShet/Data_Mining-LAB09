
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input


def cargar_datos_clasificacion() -> tuple[np.ndarray, np.ndarray]:
    """
    Carga el dataset de cáncer de mama desde scikit-learn y lo convierte a arrays.

    Returns:
        X: Array con características.
        y: Array con etiquetas binarias (1: maligno, 0: benigno).
    """
    datos = load_breast_cancer()
    X = datos.data
    y = np.where(datos.target == 0, 1, 0)
    return X, y


def crear_modelo_clasificacion() -> MLPClassifier:
    """
    Crea un modelo MLPClassifier configurado para clasificación binaria.

    Returns:
        Modelo MLPClassifier listo para entrenamiento.
    """
    return MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )


def evaluar_modelo_clasificacion(modelo: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evalúa el modelo MLP con métricas estándar y matriz de confusión.

    Args:
        modelo: Modelo entrenado.
        X_test: Datos de entrada de prueba.
        y_test: Etiquetas verdaderas.
    """
    y_pred = modelo.predict(X_test)
    print("\n🔍 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Benigno', 'Maligno'])
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.show()


def preprocesar_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga y preprocesa el dataset MNIST para su uso en CNN.

    Returns:
        X_train, X_test, y_train, y_test: Datos normalizados y etiquetas codificadas.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, X_test, y_train, y_test


def crear_cnn() -> Sequential:
    """
    Crea y compila un modelo de red neuronal convolucional simple para MNIST.

    Returns:
        Modelo CNN compilado.
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def entrenar_cnn(modelo: Sequential, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10) -> None:
    """
    Entrena una red CNN y visualiza la evolución de la precisión.

    Args:
        modelo: Modelo CNN.
        X_train: Imágenes de entrenamiento.
        y_train: Etiquetas en one-hot.
        epochs: Número de épocas de entrenamiento.
    """
    historia = modelo.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=128,
        verbose=1
    )
    plt.plot(historia.history['accuracy'], label='Entrenamiento')
    plt.plot(historia.history['val_accuracy'], label='Validación')
    plt.title('Evolución de la Precisión')
    plt.xlabel('Epoca')
    plt.ylabel('Precisión')
    plt.legend()
    plt.show()
