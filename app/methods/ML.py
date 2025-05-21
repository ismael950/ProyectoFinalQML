from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

class ML_unit:
    def __init__(self, X, y, distances, model_type="random_forest"):
        """
        Inicializa la unidad de ML con datos y modelo.

        Args:
            X: matriz de entrada (Nx24)
            y: matriz de salida (Nx2)
            distances: lista de distancias (N,)
            model_type: "random_forest" o "linear"
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.distances = np.array(distances)
        self.model_type = model_type

        # División de datos
        self.X_train, self.X_test, self.y_train, self.y_test, self.d_train, self.d_test = train_test_split(
            self.X, self.y, self.distances, test_size=0.7, random_state=42
        )

        # Escalador para regresión lineal
        if model_type == "linear":
            base_model = LinearRegression()
            self.model = MultiOutputRegressor(make_pipeline(StandardScaler(), base_model))
        elif model_type == "random_forest":
            base_model = RandomForestRegressor(random_state=42)
            self.model = MultiOutputRegressor(base_model)
        else:
            raise ValueError("Modelo no soportado. Usa 'random_forest' o 'linear'.")

    def train(self):
        """Entrena el modelo."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evalúa y reporta R² para ΔE1 y ΔE2."""
        y_pred = self.model.predict(self.X_test)
        score = r2_score(self.y_test, y_pred, multioutput='raw_values')
        print(f"R² scores: ΔE1 = {score[0]:.4f}, ΔE2 = {score[1]:.4f}")
        return score

    def predict(self, X_input):
        """Predice ΔE1 y ΔE2."""
        return self.model.predict(np.array(X_input))

    def visualize_vs_distance(self, save_path="static/images/graph.png"):
        """Grafica ΔE vs distancia para datos reales y predichos y guarda la imagen."""
        y_pred = self.model.predict(self.X_test)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].scatter(self.d_test, self.y_test[:, 0], label='ΔE1 real', alpha=0.5)
        axs[0].scatter(self.d_test, y_pred[:, 0], label='ΔE1 predicho', alpha=0.5)
        axs[0].set_title("Distancia vs ΔE1")
        axs[0].set_xlabel("Distancia")
        axs[0].set_ylabel("ΔE1")
        axs[0].legend()

        axs[1].scatter(self.d_test, self.y_test[:, 1], label='ΔE2 real', alpha=0.5)
        axs[1].scatter(self.d_test, y_pred[:, 1], label='ΔE2 predicho', alpha=0.5)
        axs[1].set_title("Distancia vs ΔE2")
        axs[1].set_xlabel("Distancia")
        axs[1].set_ylabel("ΔE2")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(save_path)  # ⬅ Guarda la imagen
        plt.close()  # ⬅ Cierra la figura para evitar sobrecarga

