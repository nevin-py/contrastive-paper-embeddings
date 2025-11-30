"""Evaluation and clustering module for paper embeddings."""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config import EvaluationConfig
from model import ContrastivePaperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """Evaluator for clustering quality of learned embeddings."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.plot_dir = Path(config.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        method: str = "kmeans"
    ) -> np.ndarray:
        """
        Cluster embeddings using specified method.
        
        Args:
            embeddings: Embedding matrix [n_samples, embedding_dim]
            n_clusters: Number of clusters
            method: Clustering method ("kmeans", "spectral", "agglomerative")
            
        Returns:
            Cluster labels
        """
        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
        elif method == "spectral":
            clusterer = SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                affinity='nearest_neighbors',
                n_neighbors=10
            )
        elif method == "agglomerative":
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        labels = clusterer.fit_predict(embeddings)
        return labels
    
    def compute_metrics(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted cluster labels
            embeddings: Optional embeddings for silhouette score
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if "nmi" in self.config.metrics:
            metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
            
        if "ari" in self.config.metrics:
            metrics['ari'] = adjusted_rand_score(true_labels, pred_labels)
            
        if "homogeneity" in self.config.metrics:
            metrics['homogeneity'] = homogeneity_score(true_labels, pred_labels)
            
        if "completeness" in self.config.metrics:
            metrics['completeness'] = completeness_score(true_labels, pred_labels)
            
        if "v_measure" in self.config.metrics or True:
            metrics['v_measure'] = v_measure_score(true_labels, pred_labels)
            
        if "silhouette" in self.config.metrics and embeddings is not None:
            # Subsample for large datasets
            if len(embeddings) > 10000:
                idx = np.random.choice(len(embeddings), 10000, replace=False)
                metrics['silhouette'] = silhouette_score(
                    embeddings[idx], pred_labels[idx]
                )
            else:
                metrics['silhouette'] = silhouette_score(embeddings, pred_labels)
                
        return metrics
    
    def reduce_dimensionality(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce embedding dimensionality for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Reduction method ("umap", "tsne", "pca")
            n_components: Target dimensions
            
        Returns:
            Low-dimensional embeddings
        """
        if method == "umap" and HAS_UMAP:
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=30,
                n_iter=1000
            )
        else:  # PCA as fallback
            reducer = PCA(n_components=n_components, random_state=42)
            
        return reducer.fit_transform(embeddings)
    
    def plot_clusters(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        title: str = "Cluster Visualization",
        label_names: Optional[Dict[int, str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot 2D cluster visualization.
        
        Args:
            embeddings_2d: 2D embeddings
            labels: Cluster or category labels
            title: Plot title
            label_names: Mapping from label indices to names
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 10))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names.get(label, f"Cluster {label}") if label_names else f"Cluster {label}"
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx]],
                label=name,
                alpha=0.6,
                s=20
            )
            
        plt.title(title, fontsize=14)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        plt.close()
        
    def plot_confusion_matrix(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        label_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix between true and predicted labels."""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_names if label_names else np.unique(pred_labels),
            yticklabels=label_names if label_names else np.unique(true_labels)
        )
        plt.title("Confusion Matrix: True Categories vs Predicted Clusters")
        plt.xlabel("Predicted Cluster")
        plt.ylabel("True Category")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()
        
    def evaluate(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        category_names: Optional[Dict[int, str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Full evaluation pipeline.
        
        Args:
            embeddings: Paper embeddings
            true_labels: Ground truth category labels
            category_names: Mapping from label indices to category names
            
        Returns:
            Dictionary of metrics for each clustering method
        """
        n_clusters = self.config.num_clusters or len(np.unique(true_labels))
        results = {}
        
        logger.info(f"Evaluating with {n_clusters} clusters...")
        
        # Try each clustering method
        for method in self.config.clustering_methods:
            logger.info(f"Clustering with {method}...")
            
            pred_labels = self.cluster_embeddings(embeddings, n_clusters, method)
            metrics = self.compute_metrics(true_labels, pred_labels, embeddings)
            
            results[method] = metrics
            
            logger.info(f"  {method}: NMI={metrics.get('nmi', 0):.4f}, "
                       f"ARI={metrics.get('ari', 0):.4f}, "
                       f"V-measure={metrics.get('v_measure', 0):.4f}")
            
        # Dimensionality reduction and visualization
        logger.info(f"Reducing dimensions with {self.config.visualization_method}...")
        embeddings_2d = self.reduce_dimensionality(
            embeddings,
            method=self.config.visualization_method
        )
        
        # Plot true categories
        self.plot_clusters(
            embeddings_2d,
            true_labels,
            title="Paper Embeddings by True Category",
            label_names=category_names,
            save_path=self.plot_dir / "clusters_true_categories.png"
        )
        
        # Plot predicted clusters (using best method by NMI)
        best_method = max(results.keys(), key=lambda m: results[m].get('nmi', 0))
        best_pred_labels = self.cluster_embeddings(embeddings, n_clusters, best_method)
        
        self.plot_clusters(
            embeddings_2d,
            best_pred_labels,
            title=f"Paper Embeddings by Predicted Cluster ({best_method})",
            save_path=self.plot_dir / f"clusters_predicted_{best_method}.png"
        )
        
        # Confusion matrix
        label_names_list = [category_names.get(i, f"Cat {i}") 
                          for i in range(n_clusters)] if category_names else None
        self.plot_confusion_matrix(
            true_labels,
            best_pred_labels,
            label_names=label_names_list,
            save_path=self.plot_dir / "confusion_matrix.png"
        )
        
        return results


class BaselineComparison:
    """Compare learned embeddings with baseline methods."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluator = ClusteringEvaluator(config)
        
    def get_tfidf_embeddings(
        self,
        texts: List[str],
        max_features: int = 5000
    ) -> np.ndarray:
        """Get TF-IDF embeddings as baseline."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings
    
    def get_sentence_transformer_embeddings(
        self,
        texts: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32
    ) -> np.ndarray:
        """Get pre-trained sentence transformer embeddings."""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )
        return embeddings
    
    def compare_all(
        self,
        texts: List[str],
        true_labels: np.ndarray,
        learned_embeddings: np.ndarray,
        category_names: Optional[Dict[int, str]] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare all embedding methods.
        
        Args:
            texts: Paper texts
            true_labels: Ground truth labels
            learned_embeddings: Embeddings from trained contrastive model
            category_names: Category name mapping
            
        Returns:
            Comparison results
        """
        results = {}
        
        # Learned embeddings
        logger.info("Evaluating learned embeddings...")
        results['contrastive'] = self.evaluator.evaluate(
            learned_embeddings, true_labels, category_names
        )
        
        # TF-IDF baseline
        logger.info("Evaluating TF-IDF baseline...")
        tfidf_embeddings = self.get_tfidf_embeddings(texts)
        results['tfidf'] = self.evaluator.evaluate(
            tfidf_embeddings, true_labels, category_names
        )
        
        # Pre-trained sentence transformers
        logger.info("Evaluating pre-trained sentence transformer...")
        st_embeddings = self.get_sentence_transformer_embeddings(texts)
        results['pretrained_st'] = self.evaluator.evaluate(
            st_embeddings, true_labels, category_names
        )
        
        # Print comparison table
        self._print_comparison_table(results)
        
        return results
    
    def _print_comparison_table(self, results: Dict):
        """Print formatted comparison table."""
        print("\n" + "="*80)
        print("EMBEDDING COMPARISON RESULTS")
        print("="*80)
        
        methods = list(results.keys())
        clustering_methods = list(results[methods[0]].keys())
        
        for cluster_method in clustering_methods:
            print(f"\nClustering: {cluster_method.upper()}")
            print("-" * 60)
            print(f"{'Method':<20} {'NMI':>10} {'ARI':>10} {'V-measure':>12}")
            print("-" * 60)
            
            for method in methods:
                metrics = results[method].get(cluster_method, {})
                nmi = metrics.get('nmi', 0)
                ari = metrics.get('ari', 0)
                vm = metrics.get('v_measure', 0)
                print(f"{method:<20} {nmi:>10.4f} {ari:>10.4f} {vm:>12.4f}")
                
        print("="*80 + "\n")


def evaluate_model(
    model: ContrastivePaperModel,
    texts: List[str],
    true_labels: np.ndarray,
    category_names: Optional[Dict[int, str]] = None,
    config: Optional[EvaluationConfig] = None,
    device: str = "cuda",
    compare_baselines: bool = True
) -> Dict:
    """
    Evaluate trained model and optionally compare with baselines.
    
    Args:
        model: Trained contrastive model
        texts: Paper texts
        true_labels: Ground truth labels
        category_names: Category name mapping
        config: Evaluation configuration
        device: Device for inference
        compare_baselines: Whether to compare with baseline methods
        
    Returns:
        Evaluation results
    """
    if config is None:
        config = EvaluationConfig()
        
    # Get learned embeddings
    logger.info("Extracting learned embeddings...")
    model.to(device)
    learned_embeddings = model.encode_texts(
        texts,
        device=device,
        use_projection=False  # Use encoder embeddings, not projection
    ).numpy()
    
    if compare_baselines:
        comparison = BaselineComparison(config)
        results = comparison.compare_all(
            texts, true_labels, learned_embeddings, category_names
        )
    else:
        evaluator = ClusteringEvaluator(config)
        results = {'contrastive': evaluator.evaluate(
            learned_embeddings, true_labels, category_names
        )}
        
    return results


if __name__ == "__main__":
    # Test evaluation with dummy data
    config = EvaluationConfig()
    evaluator = ClusteringEvaluator(config)
    
    # Create dummy embeddings
    np.random.seed(42)
    n_samples = 300
    n_clusters = 3
    
    # Generate clustered data
    embeddings = np.vstack([
        np.random.randn(100, 128) + np.array([2] * 128),
        np.random.randn(100, 128) + np.array([-2] * 128),
        np.random.randn(100, 128) + np.array([0] * 128),
    ])
    
    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    category_names = {0: "Computer Vision", 1: "NLP", 2: "Reinforcement Learning"}
    
    results = evaluator.evaluate(embeddings, true_labels, category_names)
    print("Evaluation results:", results)
