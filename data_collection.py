"""Data collection module for ArXiv papers."""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import arxiv
import pandas as pd
from tqdm import tqdm

from config import DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivCollector:
    """Collects papers from ArXiv API."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_papers_by_category(
        self, 
        category: str, 
        max_results: int
    ) -> List[Dict]:
        """Fetch papers from a specific ArXiv category."""
        papers = []
        
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )
        
        try:
            for result in tqdm(
                client.results(search), 
                total=max_results, 
                desc=f"Fetching {category}"
            ):
                abstract = result.summary.replace('\n', ' ').strip()
                
                # Filter by abstract length
                if len(abstract) < self.config.min_abstract_length:
                    continue
                if len(abstract) > self.config.max_abstract_length:
                    abstract = abstract[:self.config.max_abstract_length]
                
                paper = {
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'title': result.title.replace('\n', ' ').strip(),
                    'abstract': abstract,
                    'primary_category': result.primary_category,
                    'categories': ','.join(result.categories),
                    'published': result.published.isoformat() if result.published else None,
                    'authors': ','.join([a.name for a in result.authors[:10]]),
                    'url': result.entry_id,
                }
                papers.append(paper)
                
                if len(papers) >= max_results:
                    break
                    
        except Exception as e:
            logger.error(f"Error fetching papers for {category}: {e}")
            
        return papers
    
    def collect_all_categories(self) -> pd.DataFrame:
        """Collect papers from all configured categories."""
        all_papers = []
        
        for category in self.config.categories:
            logger.info(f"Collecting papers from {category}...")
            papers = self.fetch_papers_by_category(
                category, 
                self.config.max_papers_per_category
            )
            all_papers.extend(papers)
            
            # Rate limiting
            time.sleep(2)
        
        df = pd.DataFrame(all_papers)
        
        # Remove duplicates based on arxiv_id
        df = df.drop_duplicates(subset=['arxiv_id'], keep='first')
        
        logger.info(f"Collected {len(df)} unique papers")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Save collected data to disk."""
        if filename is None:
            filename = self.config.cache_file
        
        filepath = self.data_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} papers to {filepath}")
        
    def load_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Load data from disk."""
        if filename is None:
            filename = self.config.cache_file
            
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} papers from {filepath}")
        return df
    
    def get_or_collect_data(self) -> pd.DataFrame:
        """Get data from cache or collect if not available."""
        try:
            return self.load_data()
        except FileNotFoundError:
            logger.info("Cache not found, collecting data...")
            df = self.collect_all_categories()
            self.save_data(df)
            return df


class PaperDataset:
    """Dataset wrapper for paper data with preprocessing."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.papers = self._preprocess()
        
    def _preprocess(self) -> List[Dict]:
        """Preprocess papers for training."""
        papers = []
        for idx, row in self.df.iterrows():
            papers.append({
                'id': idx,
                'arxiv_id': row['arxiv_id'],
                'text': f"{row['title']}. {row['abstract']}",
                'title': row['title'],
                'abstract': row['abstract'],
                'category': row['primary_category'],
            })
        return papers
    
    def __len__(self):
        return len(self.papers)
    
    def __getitem__(self, idx):
        return self.papers[idx]
    
    def get_texts(self) -> List[str]:
        """Get all paper texts."""
        return [p['text'] for p in self.papers]
    
    def get_categories(self) -> List[str]:
        """Get all paper categories."""
        return [p['category'] for p in self.papers]
    
    def get_category_labels(self) -> Tuple[List[int], Dict[str, int]]:
        """Get numeric category labels and mapping."""
        categories = self.get_categories()
        unique_categories = sorted(set(categories))
        category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        labels = [category_to_idx[cat] for cat in categories]
        return labels, category_to_idx


if __name__ == "__main__":
    # Test data collection
    config = DataConfig(max_papers_per_category=50)  # Small test
    collector = ArxivCollector(config)
    df = collector.get_or_collect_data()
    print(df.head())
    print(f"\nCategory distribution:\n{df['primary_category'].value_counts()}")
