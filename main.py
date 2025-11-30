"""Main entry point for contrastive learning on scientific papers."""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

from config import Config, DataConfig, ModelConfig, TrainingConfig, EvaluationConfig
from data_collection import ArxivCollector, PaperDataset
from model import ContrastivePaperModel
from train import Trainer, set_seed
from evaluation import evaluate_model, ClusteringEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_data(config: Config) -> PaperDataset:
    """Collect or load ArXiv papers."""
    collector = ArxivCollector(config.data)
    df = collector.get_or_collect_data()
    dataset = PaperDataset(df)
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Categories: {df['primary_category'].unique().tolist()}")
    
    return dataset


def train(config: Config, dataset: PaperDataset) -> ContrastivePaperModel:
    """Train the contrastive model."""
    set_seed(config.seed)
    
    # Create model
    model = ContrastivePaperModel(config.model)
    
    # Get texts
    texts = dataset.get_texts()
    
    # Create trainer
    trainer = Trainer(model, config, texts)
    
    # Train
    trained_model = trainer.train()
    
    return trained_model


def evaluate(
    config: Config,
    model: ContrastivePaperModel,
    dataset: PaperDataset,
    compare_baselines: bool = True
):
    """Evaluate the trained model."""
    texts = dataset.get_texts()
    labels, category_to_idx = dataset.get_category_labels()
    labels = np.array(labels)
    
    # Invert mapping for display
    idx_to_category = {v: k for k, v in category_to_idx.items()}
    
    device = config.training.device if torch.cuda.is_available() else "cpu"
    
    results = evaluate_model(
        model=model,
        texts=texts,
        true_labels=labels,
        category_names=idx_to_category,
        config=config.evaluation,
        device=device,
        compare_baselines=compare_baselines
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive Learning for Scientific Paper Embeddings"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "train", "evaluate", "full"],
        default="full",
        help="Mode: collect data, train, evaluate, or full pipeline"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation"
    )
    
    # Data arguments
    parser.add_argument(
        "--max-papers",
        type=int,
        default=500,
        help="Maximum papers per category"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Contrastive loss temperature"
    )
    
    # Model arguments
    parser.add_argument(
        "--encoder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Encoder model name"
    )
    
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=128,
        help="Projection head output dimension"
    )
    
    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline comparisons in evaluation"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        data=DataConfig(
            max_papers_per_category=args.max_papers,
            data_dir=args.data_dir
        ),
        model=ModelConfig(
            encoder_name=args.encoder,
            projection_dim=args.projection_dim
        ),
        training=TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            temperature=args.temperature,
            device=args.device,
            use_wandb=args.wandb
        ),
        seed=args.seed
    )
    
    logger.info(f"Running in {args.mode} mode")
    
    if args.mode == "collect":
        # Just collect data
        dataset = collect_data(config)
        logger.info("Data collection complete!")
        
    elif args.mode == "train":
        # Load data and train
        dataset = collect_data(config)
        model = train(config, dataset)
        logger.info("Training complete!")
        
    elif args.mode == "evaluate":
        # Load checkpoint and evaluate
        if args.checkpoint is None:
            args.checkpoint = "checkpoints/best_model.pt"
            
        dataset = collect_data(config)
        
        # Load model
        model = ContrastivePaperModel(config.model)
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        results = evaluate(config, model, dataset, not args.no_baselines)
        logger.info("Evaluation complete!")
        
    elif args.mode == "full":
        # Full pipeline
        logger.info("=== Step 1: Data Collection ===")
        dataset = collect_data(config)
        
        logger.info("=== Step 2: Training ===")
        model = train(config, dataset)
        
        logger.info("=== Step 3: Evaluation ===")
        results = evaluate(config, model, dataset, not args.no_baselines)
        
        logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
