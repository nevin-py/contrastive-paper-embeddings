"""SimCLR-style contrastive learning model for paper embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from transformers import AutoModel, AutoTokenizer

from config import ModelConfig


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TextEncoder(nn.Module):
    """Transformer-based text encoder."""
    
    def __init__(
        self,
        model_name: str,
        pooling_strategy: str = "mean",
        freeze_layers: int = 0
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        
        # Freeze early layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
            
    def _freeze_layers(self, num_layers: int):
        """Freeze the first n encoder layers."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze encoder layers
        if hasattr(self.encoder, 'encoder'):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, 'layers'):
            layers = self.encoder.layers
        else:
            return
            
        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    def pool(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool hidden states to get sentence embedding."""
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0]
        elif self.pooling_strategy == "max":
            # Mask padding tokens
            hidden_states[attention_mask == 0] = -1e9
            return hidden_states.max(dim=1)[0]
        else:  # mean pooling
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
            
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return self.pool(outputs.last_hidden_state, attention_mask)
    
    def encode(
        self, 
        texts: list,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Encode a list of texts to embeddings."""
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                embeddings = self(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)


class ContrastivePaperModel(nn.Module):
    """SimCLR-style contrastive learning model for papers."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Text encoder
        self.encoder = TextEncoder(
            model_name=config.encoder_name,
            pooling_strategy=config.pooling_strategy,
            freeze_layers=config.freeze_encoder_layers
        )
        
        # Projection head
        self.projection = ProjectionHead(
            input_dim=config.encoder_dim,
            hidden_dim=config.projection_hidden_dim,
            output_dim=config.projection_dim,
            num_layers=config.num_projection_layers
        )
        
        # Store tokenizer reference
        self.tokenizer = self.encoder.tokenizer
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embedding: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            return_embedding: Whether to return the encoder embedding (before projection)
            
        Returns:
            projection: Projected embedding for contrastive loss
            embedding: Encoder embedding (if return_embedding=True)
        """
        embedding = self.encoder(input_ids, attention_mask)
        projection = self.projection(embedding)
        
        # L2 normalize projections for contrastive learning
        projection = F.normalize(projection, dim=1)
        
        if return_embedding:
            return projection, embedding
        return projection, None
    
    def encode_texts(
        self,
        texts: list,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 32,
        use_projection: bool = False
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            device: Device to use
            max_length: Maximum token length
            batch_size: Batch size
            use_projection: Whether to use projected embeddings
            
        Returns:
            Tensor of embeddings
        """
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                projection, embedding = self(
                    input_ids, 
                    attention_mask,
                    return_embedding=True
                )
                
                if use_projection:
                    all_embeddings.append(projection.cpu())
                else:
                    all_embeddings.append(embedding.cpu())
                    
        return torch.cat(all_embeddings, dim=0)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        z_i: torch.Tensor, 
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z_i: Embeddings from first augmented view [batch_size, embedding_dim]
            z_j: Embeddings from second augmented view [batch_size, embedding_dim]
            
        Returns:
            Contrastive loss
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, embedding_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create mask for positive pairs
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(device)
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE loss with hard negative mining option."""
    
    def __init__(
        self, 
        temperature: float = 0.07,
        hard_negative_weight: float = 0.0
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            z_i: Anchor embeddings
            z_j: Positive embeddings
            hard_negatives: Optional hard negative embeddings
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Positive similarities
        pos_sim = torch.sum(z_i * z_j, dim=1, keepdim=True) / self.temperature
        
        # Negative similarities (all other samples in batch)
        neg_sim = torch.mm(z_i, z_j.t()) / self.temperature
        
        # Mask out positive pairs from negatives
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        
        # Add hard negatives if provided
        if hard_negatives is not None and self.hard_negative_weight > 0:
            hard_neg_sim = torch.mm(z_i, hard_negatives.t()) / self.temperature
            neg_sim = torch.cat([neg_sim, hard_neg_sim * self.hard_negative_weight], dim=1)
        
        # Combine for softmax
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


if __name__ == "__main__":
    # Test model
    config = ModelConfig()
    model = ContrastivePaperModel(config)
    
    # Test with dummy data
    texts = [
        "Deep learning for computer vision applications.",
        "Natural language processing with transformers.",
        "Reinforcement learning for robotics."
    ]
    
    tokenizer = model.tokenizer
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Forward pass
    projection, embedding = model(
        encoded['input_ids'],
        encoded['attention_mask'],
        return_embedding=True
    )
    
    print(f"Input shape: {encoded['input_ids'].shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Projection shape: {projection.shape}")
    
    # Test NT-Xent loss
    loss_fn = NTXentLoss(temperature=0.07)
    z_i = torch.randn(32, 128)
    z_j = torch.randn(32, 128)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    loss = loss_fn(z_i, z_j)
    print(f"NT-Xent loss: {loss.item():.4f}")
