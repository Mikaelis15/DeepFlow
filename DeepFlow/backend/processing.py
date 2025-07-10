import time
import os
import torch
import pandas as pd
import numpy as np
from joblib import load as joblib_load, dump as joblib_dump
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
from sklearn.preprocessing import StandardScaler
import tempfile
import config
from backend.models import AutoEncoder, IDEC, VAE, VaDE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import umap
class IDECWorker(QObject):
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True
        self.status_update.emit("Stopping analysis...")

    def export_results_to_csv(self, original_df, y_pred, rbc_count, plt_count):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as temp_f:
            temp_csv_path = temp_f.name

        y_pred_np = y_pred.cpu().numpy()
        rbc_probs = y_pred_np[:, 0]

        sorted_indices = np.argsort(-rbc_probs)
        cluster_assignments = np.zeros(len(rbc_probs), dtype=int)
        cluster_assignments[sorted_indices[:rbc_count]] = 1

        results_df = original_df.copy()
        results_df['cluster_id'] = cluster_assignments
        results_df['RBC_prob'] = y_pred_np[:, 0]
        results_df['PLT_prob'] = y_pred_np[:, 1]
        results_df['predicted_cell_type'] = results_df['cluster_id'].map({1: 'RBC', 0: 'PLT'})

        results_df.to_csv(temp_csv_path, index=False)
        return temp_csv_path

    def run(self):
        try:
            self.status_update.emit("Loading data from CSV...")
            if self._is_stopped: return
            df = pd.read_csv(self.csv_path)
            features = df.iloc[:, :config.INPUT_DIM].values.astype(np.float32)

            if not os.path.exists(config.SCALER_PATH):
                self.status_update.emit("Calibrating new scaler...")
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                joblib_dump(scaler, config.SCALER_PATH)
                self.status_update.emit("New scaler saved!")
            else:
                self.status_update.emit("Loading existing scaler...")
                scaler = joblib_load(config.SCALER_PATH)
                scaled_features = scaler.transform(features)

            self.status_update.emit("Loading analysis models...")
            autoencoder = AutoEncoder(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM)
            autoencoder.load_state_dict(torch.load(config.AUTOENCODER_MODEL_PATH, map_location=config.DEVICE))
            idec_model = IDEC(autoencoder=autoencoder, n_clusters=config.N_CLUSTERS, latent_dim=config.LATENT_DIM)
            idec_model.load_state_dict(torch.load(config.IDEC_MODEL_PATH, map_location=config.DEVICE))
            idec_model.to(config.DEVICE)
            idec_model.eval()

            features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(config.DEVICE)

            self.status_update.emit("Running cell analysis...")
            if self._is_stopped: return
            with torch.no_grad():
                _x_recon, y_pred, q, z = idec_model(features_tensor)

            y_pred_sum = y_pred.sum(dim=1, keepdim=True) + 1e-8
            y_pred_norm = y_pred / y_pred_sum
            rbc_ratio_sum = y_pred_norm[:, 0].sum().item()
            plt_ratio_sum = y_pred_norm[:, 1].sum().item()
            ratio_total = rbc_ratio_sum + plt_ratio_sum + 1e-8
            rbc_ratio = rbc_ratio_sum / ratio_total
            plt_ratio = plt_ratio_sum / ratio_total
            total_cells = features_tensor.shape[0]
            rbc_count = int(rbc_ratio * total_cells)
            plt_count = int(plt_ratio * total_cells)

            self.status_update.emit("Generating cluster plot...")
            if self._is_stopped: return
            cluster_assignments = torch.argmax(q, dim=1).cpu().numpy()
            plot_path = self.create_cluster_plot(z.cpu().numpy(), cluster_assignments)

            self.status_update.emit("Exporting cell classifications...")
            temp_csv_path = self.export_results_to_csv(df, y_pred, rbc_count, plt_count)

            self.status_update.emit("Analysis Complete!")
            time.sleep(1)

            results = (rbc_count, plt_count, plot_path, temp_csv_path)
            self.finished.emit(results)

        except FileNotFoundError as e:
            if not self._is_stopped:
                self.error.emit(f"A required file was not found: {e.filename}")
        except RuntimeError as e:
            if not self._is_stopped:
                self.error.emit(f"Model loading error: {str(e)}")
        except Exception as e:
            if not self._is_stopped:
                self.error.emit(f"An analysis error occurred: {str(e)}")

    def create_cluster_plot(self, latent_vectors, clusters):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 6))
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(latent_vectors)
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=clusters, cmap='viridis', s=5, alpha=0.7)
        ax.set_title("Cell Clusters (PCA Visualization)", color="#e0e0e0")
        ax.set_xlabel("Principal Component 1", color="#cccccc")
        ax.set_ylabel("Principal Component 2", color="#cccccc")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color="#555555")
        legend_elements = scatter.legend_elements()
        legend_labels = ["PLT", "RBC"]
        if len(legend_elements[0]) == len(legend_labels):
             ax.legend(legend_elements[0], legend_labels, title="Cell Types")
        plot_path = os.path.join(config.OUTPUT_DIR, "cluster_plot.png")
        fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return plot_path

class VaDEWorker(QObject):
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    plot_generated = pyqtSignal(str, str)

    def __init__(self, idec_output_csv_path):
        super().__init__()
        self.idec_output_csv_path = idec_output_csv_path
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True
        self.status_update.emit("Stopping VaDE analysis...")

    def run(self):
        try:
            self.status_update.emit("Loading IDEC output CSV...")
            if self._is_stopped: return
            df = pd.read_csv(self.idec_output_csv_path)

            if 'predicted_cell_type' not in df.columns:
                raise ValueError("Missing 'predicted_cell_type' in CSV.")

            rbc_df = df[df['predicted_cell_type'] == 'RBC'].copy()
            if rbc_df.empty:
                raise ValueError("No RBC rows found for VaDE analysis.")

            features = rbc_df.drop(columns=['ID', 'cluster_id', 'RBC_prob', 'PLT_prob', 'predicted_cell_type'], errors='ignore')
            features = features.iloc[:, :config.INPUT_DIM].values.astype(np.float32)

            self.status_update.emit("Loading scaler...")
            scaler = joblib_load(config.SCALER_PATH)
            scaled_features = scaler.transform(features)

            self.status_update.emit("Loading VaDE model...")
            vade_model = VaDE(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM, n_clusters=config.VADE_N_CLUSTERS)
            vade_model.load_state_dict(torch.load(config.VADE_MODEL_PATH, map_location=config.DEVICE))
            vade_model.to(config.DEVICE)
            vade_model.eval()

            features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(config.DEVICE)

            self.status_update.emit("Running VaDE clustering...")
            if self._is_stopped: return
            with torch.no_grad():
                x_recon, mu, logvar, z = vade_model(features_tensor)
                gamma = vade_model.compute_gamma(z)
                pred_clusters = torch.argmax(gamma, dim=1)

            pred_clusters_np = pred_clusters.cpu().numpy()
            subtype_counts = np.bincount(pred_clusters_np, minlength=config.VADE_N_CLUSTERS)

            self.status_update.emit("Generating VaDE cluster plot...")
            if self._is_stopped: return
            
            plot_path, plot_error_message = self.create_cluster_plot(z.cpu().numpy())
            self.plot_generated.emit(plot_path, plot_error_message)

            if plot_error_message:
                self.error.emit(f"VaDE plot generation failed: {plot_error_message}")
                return

            time.sleep(1)
            self.status_update.emit("VaDE Analysis Complete!")
            self.finished.emit((*subtype_counts, plot_path))

        except FileNotFoundError as e:
            if not self._is_stopped:
                self.error.emit(f"Missing file: {e.filename}")
        except RuntimeError as e:
            if not self._is_stopped:
                self.error.emit(f"Model loading error: {str(e)}")
        except Exception as e:
            if not self._is_stopped:
                self.error.emit(f"VaDE analysis error: {str(e)}")

    import time
import os
import torch
import pandas as pd
import numpy as np
from joblib import load as joblib_load, dump as joblib_dump
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
from sklearn.preprocessing import StandardScaler
import tempfile
import config
from backend.models import AutoEncoder, IDEC, VAE, VaDE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import umap
class IDECWorker(QObject):
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True
        self.status_update.emit("Stopping analysis...")

    def export_results_to_csv(self, original_df, y_pred, rbc_count, plt_count):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as temp_f:
            temp_csv_path = temp_f.name

        y_pred_np = y_pred.cpu().numpy()
        rbc_probs = y_pred_np[:, 0]

        sorted_indices = np.argsort(-rbc_probs)
        cluster_assignments = np.zeros(len(rbc_probs), dtype=int)
        cluster_assignments[sorted_indices[:rbc_count]] = 1

        results_df = original_df.copy()
        results_df['cluster_id'] = cluster_assignments
        results_df['RBC_prob'] = y_pred_np[:, 0]
        results_df['PLT_prob'] = y_pred_np[:, 1]
        results_df['predicted_cell_type'] = results_df['cluster_id'].map({1: 'RBC', 0: 'PLT'})

        results_df.to_csv(temp_csv_path, index=False)
        return temp_csv_path

    def run(self):
        try:
            self.status_update.emit("Loading data from CSV...")
            if self._is_stopped: return
            df = pd.read_csv(self.csv_path)
            features = df.iloc[:, :config.INPUT_DIM].values.astype(np.float32)

            if not os.path.exists(config.SCALER_PATH):
                self.status_update.emit("Calibrating new scaler...")
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                joblib_dump(scaler, config.SCALER_PATH)
                self.status_update.emit("New scaler saved!")
            else:
                self.status_update.emit("Loading existing scaler...")
                scaler = joblib_load(config.SCALER_PATH)
                scaled_features = scaler.transform(features)

            self.status_update.emit("Loading analysis models...")
            autoencoder = AutoEncoder(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM)
            autoencoder.load_state_dict(torch.load(config.AUTOENCODER_MODEL_PATH, map_location=config.DEVICE))
            idec_model = IDEC(autoencoder=autoencoder, n_clusters=config.N_CLUSTERS, latent_dim=config.LATENT_DIM)
            idec_model.load_state_dict(torch.load(config.IDEC_MODEL_PATH, map_location=config.DEVICE))
            idec_model.to(config.DEVICE)
            idec_model.eval()

            features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(config.DEVICE)

            self.status_update.emit("Running cell analysis...")
            if self._is_stopped: return
            with torch.no_grad():
                _x_recon, y_pred, q, z = idec_model(features_tensor)

            y_pred_sum = y_pred.sum(dim=1, keepdim=True) + 1e-8
            y_pred_norm = y_pred / y_pred_sum
            rbc_ratio_sum = y_pred_norm[:, 0].sum().item()
            plt_ratio_sum = y_pred_norm[:, 1].sum().item()
            ratio_total = rbc_ratio_sum + plt_ratio_sum + 1e-8
            rbc_ratio = rbc_ratio_sum / ratio_total
            plt_ratio = plt_ratio_sum / ratio_total
            total_cells = features_tensor.shape[0]
            rbc_count = int(rbc_ratio * total_cells)
            plt_count = int(plt_ratio * total_cells)

            self.status_update.emit("Generating cluster plot...")
            if self._is_stopped: return
            cluster_assignments = torch.argmax(q, dim=1).cpu().numpy()
            plot_path = self.create_cluster_plot(z.cpu().numpy(), cluster_assignments)

            self.status_update.emit("Exporting cell classifications...")
            temp_csv_path = self.export_results_to_csv(df, y_pred, rbc_count, plt_count)

            self.status_update.emit("Analysis Complete!")
            time.sleep(1)

            results = (rbc_count, plt_count, plot_path, temp_csv_path)
            self.finished.emit(results)

        except FileNotFoundError as e:
            if not self._is_stopped:
                self.error.emit(f"A required file was not found: {e.filename}")
        except RuntimeError as e:
            if not self._is_stopped:
                self.error.emit(f"Model loading error: {str(e)}")
        except Exception as e:
            if not self._is_stopped:
                self.error.emit(f"An analysis error occurred: {str(e)}")

    def create_cluster_plot(self, latent_vectors, clusters):
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(6, 6))
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(latent_vectors)
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=clusters, cmap='viridis', s=5, alpha=0.7)
        ax.set_title("Cell Clusters (PCA Visualization)", color="#422065")
        ax.set_xlabel("Principal Component 1", color="#422065")
        ax.set_ylabel("Principal Component 2", color="#422065")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color="#3C3344")
        legend_elements = scatter.legend_elements()
        legend_labels = ["PLT", "RBC"]
        if len(legend_elements[0]) == len(legend_labels):
             ax.legend(legend_elements[0], legend_labels, title="Cell Types")
        plot_path = os.path.join(config.OUTPUT_DIR, "cluster_plot.png")
        fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return plot_path

class VaDEWorker(QObject):
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    plot_generated = pyqtSignal(str, str)

    def __init__(self, idec_output_csv_path):
        super().__init__()
        self.idec_output_csv_path = idec_output_csv_path
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True
        self.status_update.emit("Stopping VaDE analysis...")

    def run(self):
        try:
            self.status_update.emit("Loading IDEC output CSV...")
            if self._is_stopped: return
            df = pd.read_csv(self.idec_output_csv_path)

            if 'predicted_cell_type' not in df.columns:
                raise ValueError("Missing 'predicted_cell_type' in CSV.")

            rbc_df = df[df['predicted_cell_type'] == 'RBC'].copy()
            if rbc_df.empty:
                raise ValueError("No RBC rows found for VaDE analysis.")

            features = rbc_df.drop(columns=['ID', 'cluster_id', 'RBC_prob', 'PLT_prob', 'predicted_cell_type'], errors='ignore')
            features = features.iloc[:, :config.INPUT_DIM].values.astype(np.float32)

            self.status_update.emit("Loading scaler...")
            scaler = joblib_load(config.SCALER_PATH)
            scaled_features = scaler.transform(features)

            self.status_update.emit("Loading VaDE model...")
            vade_model = VaDE(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM, n_clusters=config.VADE_N_CLUSTERS)
            vade_model.load_state_dict(torch.load(config.VADE_MODEL_PATH, map_location=config.DEVICE))
            vade_model.to(config.DEVICE)
            vade_model.eval()

            features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(config.DEVICE)

            self.status_update.emit("Running VaDE clustering...")
            if self._is_stopped: return
            with torch.no_grad():
                x_recon, mu, logvar, z = vade_model(features_tensor)
                gamma = vade_model.compute_gamma(z)
                pred_clusters = torch.argmax(gamma, dim=1)

            pred_clusters_np = pred_clusters.cpu().numpy()
            subtype_counts = np.bincount(pred_clusters_np, minlength=config.VADE_N_CLUSTERS)

            self.status_update.emit("Generating VaDE cluster plot...")
            if self._is_stopped: return
            
            plot_path, plot_error_message = self.create_cluster_plot(mu.cpu().numpy())
            self.plot_generated.emit(plot_path, plot_error_message)

            if plot_error_message:
                self.error.emit(f"VaDE plot generation failed: {plot_error_message}")
                return

            time.sleep(1)
            self.status_update.emit("VaDE Analysis Complete!")
            self.finished.emit((*subtype_counts, plot_path))

        except FileNotFoundError as e:
            if not self._is_stopped:
                self.error.emit(f"Missing file: {e.filename}")
        except RuntimeError as e:
            if not self._is_stopped:
                self.error.emit(f"Model loading error: {str(e)}")
        except Exception as e:
            if not self._is_stopped:
                self.error.emit(f"VaDE analysis error: {str(e)}")

    def create_cluster_plot(self, latent_vectors, save_path="vade_tsne_clusters.png"):
        n_points = latent_vectors.shape[0]
        max_points = 50000
        if n_points > max_points:
            idx = np.random.choice(n_points, max_points, replace=False)
            latent_vectors = latent_vectors[idx]

        plt.style.use('default')
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_coords = tsne.fit_transform(latent_vectors)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                s=15, alpha=0.6, c='#004488')

        ax.set_title("t-SNE Visualization of Latent Space")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        plot_path = os.path.join(config.OUTPUT_DIR, save_path)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return plot_path, ""