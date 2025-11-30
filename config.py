"""Configuration settings for the contrastive learning project."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data collection and processing configuration."""
    categories: List[str] = field(default_factory=lambda: [
        "cs.LG",   # Machine Learning
        "cs.AI",   # Artificial Intelligence
        "cs.CV",   # Computer Vision
        "cs.CL",   # Computation and Language (NLP)
        "cs.NE",   # Neural and Evolutionary Computing
        "stat.ML", # Statistics - Machine Learning
    ])
    max_papers_per_category: int = 500
    data_dir: str = "data"
    cache_file: str = "arxiv_papers.parquet"
    min_abstract_length: int = 100
    max_abstract_length: int = 2000


@dataclass
class AugmentationConfig:
    """Text augmentation configuration."""
    # Augmentation probabilities
    word_dropout_prob: float = 0.1
    word_shuffle_window: int = 3
    sentence_shuffle_prob: float = 0.3
    back_translation_prob: float = 0.0  # Expensive, disabled by default
    
    # Synonym replacement
    synonym_replacement_prob: float = 0.15
    max_synonyms_per_word: int = 3
    
    # Span masking
    span_mask_prob: float = 0.15
    span_mask_max_length: int = 5


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Encoder
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_dim: int = 384
    
    # Projection head
    projection_dim: int = 128
    projection_hidden_dim: int = 256
    num_projection_layers: int = 2
    
    # Pooling
    pooling_strategy: str = "mean"  # "mean", "cls", "max"
    
    # Freeze encoder layers
    freeze_encoder_layers: int = 0


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    
    # Contrastive learning
    temperature: float = 0.07
    
    # Optimization
    gradient_accumulation_steps: int = 16  # Simulate larger batches (effective 128)
    max_grad_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    
    # Logging
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "contrastive-papers"
    
    # Device
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    num_clusters: Optional[int] = None  # Auto-detect from categories
    clustering_methods: List[str] = field(default_factory=lambda: [
        "kmeans", "spectral", "agglomerative"
    ])
    
    # Visualization
    visualization_method: str = "umap"  # "umap", "tsne", "pca"
    plot_dir: str = "plots"
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "nmi", "ari", "silhouette", "homogeneity", "completeness"
    ])


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    seed: int = 42
    
    def __post_init__(self):
        if self.evaluation.num_clusters is None:
            self.evaluation.num_clusters = len(self.data.categories)
