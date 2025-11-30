"""Text augmentation strategies for contrastive learning."""

import random
import re
from typing import List, Optional, Callable
from abc import ABC, abstractmethod

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

from config import AugmentationConfig

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextAugmenter(ABC):
    """Base class for text augmentation."""
    
    @abstractmethod
    def augment(self, text: str) -> str:
        """Apply augmentation to text."""
        pass
    
    def __call__(self, text: str) -> str:
        return self.augment(text)


class WordDropout(TextAugmenter):
    """Randomly drop words from text."""
    
    def __init__(self, dropout_prob: float = 0.1):
        self.dropout_prob = dropout_prob
        
    def augment(self, text: str) -> str:
        words = text.split()
        if len(words) <= 3:
            return text
            
        kept_words = [
            word for word in words 
            if random.random() > self.dropout_prob
        ]
        
        # Ensure we keep at least 50% of words
        if len(kept_words) < len(words) * 0.5:
            kept_words = random.sample(words, max(3, len(words) // 2))
            
        return ' '.join(kept_words)


class WordShuffle(TextAugmenter):
    """Shuffle words within a local window."""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        
    def augment(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.window_size:
            return text
            
        # Shuffle within windows
        result = []
        for i in range(0, len(words), self.window_size):
            window = words[i:i + self.window_size]
            random.shuffle(window)
            result.extend(window)
            
        return ' '.join(result)


class SentenceShuffle(TextAugmenter):
    """Shuffle sentences in the text."""
    
    def __init__(self, shuffle_prob: float = 0.3):
        self.shuffle_prob = shuffle_prob
        
    def augment(self, text: str) -> str:
        if random.random() > self.shuffle_prob:
            return text
            
        sentences = sent_tokenize(text)
        if len(sentences) <= 2:
            return text
            
        # Keep first sentence, shuffle the rest
        first = sentences[0]
        rest = sentences[1:]
        random.shuffle(rest)
        
        return ' '.join([first] + rest)


class SynonymReplacement(TextAugmenter):
    """Replace words with their synonyms."""
    
    def __init__(
        self, 
        replacement_prob: float = 0.15,
        max_synonyms: int = 3
    ):
        self.replacement_prob = replacement_prob
        self.max_synonyms = max_synonyms
        
        # Words to exclude from replacement
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once'
        }
        
    def get_synonym(self, word: str) -> Optional[str]:
        """Get a random synonym for a word."""
        synsets = wordnet.synsets(word)
        if not synsets:
            return None
            
        synonyms = set()
        for synset in synsets[:self.max_synonyms]:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
                    
        if not synonyms:
            return None
            
        return random.choice(list(synonyms))
    
    def augment(self, text: str) -> str:
        words = text.split()
        result = []
        
        for word in words:
            # Skip short words, stopwords, and non-alphabetic
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if (
                len(clean_word) < 4 or 
                clean_word.lower() in self.stopwords or
                random.random() > self.replacement_prob
            ):
                result.append(word)
                continue
                
            synonym = self.get_synonym(clean_word)
            if synonym:
                # Preserve original punctuation
                if word != clean_word:
                    punct = word.replace(clean_word, '')
                    synonym = synonym + punct
                result.append(synonym)
            else:
                result.append(word)
                
        return ' '.join(result)


class SpanMasking(TextAugmenter):
    """Mask random spans with [MASK] token."""
    
    def __init__(
        self, 
        mask_prob: float = 0.15,
        max_span_length: int = 5
    ):
        self.mask_prob = mask_prob
        self.max_span_length = max_span_length
        self.mask_token = "[MASK]"
        
    def augment(self, text: str) -> str:
        words = text.split()
        if len(words) < 10:
            return text
            
        result = []
        i = 0
        
        while i < len(words):
            if random.random() < self.mask_prob:
                # Mask a span
                span_length = random.randint(1, min(self.max_span_length, len(words) - i))
                result.append(self.mask_token)
                i += span_length
            else:
                result.append(words[i])
                i += 1
                
        return ' '.join(result)


class RandomInsertion(TextAugmenter):
    """Insert random words from the text at random positions."""
    
    def __init__(self, insertion_prob: float = 0.1):
        self.insertion_prob = insertion_prob
        
    def augment(self, text: str) -> str:
        words = text.split()
        if len(words) < 5:
            return text
            
        # Get content words (longer words) for insertion
        content_words = [w for w in words if len(w) > 4]
        if not content_words:
            return text
            
        result = []
        for word in words:
            result.append(word)
            if random.random() < self.insertion_prob:
                result.append(random.choice(content_words))
                
        return ' '.join(result)


class TitleAbstractSwap(TextAugmenter):
    """Swap title and abstract order (paper-specific)."""
    
    def __init__(self, swap_prob: float = 0.5):
        self.swap_prob = swap_prob
        
    def augment(self, text: str) -> str:
        if random.random() > self.swap_prob:
            return text
            
        # Try to split at first period (title separator)
        parts = text.split('. ', 1)
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}."
        return text


class CompositeAugmenter(TextAugmenter):
    """Apply multiple augmentations in sequence."""
    
    def __init__(self, augmenters: List[TextAugmenter]):
        self.augmenters = augmenters
        
    def augment(self, text: str) -> str:
        for augmenter in self.augmenters:
            text = augmenter(text)
        return text


class RandomAugmenter(TextAugmenter):
    """Randomly select and apply one augmentation."""
    
    def __init__(self, augmenters: List[TextAugmenter]):
        self.augmenters = augmenters
        
    def augment(self, text: str) -> str:
        augmenter = random.choice(self.augmenters)
        return augmenter(text)


def create_augmenter(config: AugmentationConfig) -> TextAugmenter:
    """Create the default augmentation pipeline."""
    augmenters = [
        WordDropout(config.word_dropout_prob),
        WordShuffle(config.word_shuffle_window),
        SentenceShuffle(config.sentence_shuffle_prob),
        SynonymReplacement(
            config.synonym_replacement_prob,
            config.max_synonyms_per_word
        ),
        SpanMasking(config.span_mask_prob, config.span_mask_max_length),
    ]
    
    # Use random selection for diversity
    return RandomAugmenter(augmenters)


def create_strong_augmenter(config: AugmentationConfig) -> TextAugmenter:
    """Create a stronger augmentation pipeline (for view 2)."""
    augmenters = [
        CompositeAugmenter([
            WordDropout(config.word_dropout_prob * 1.5),
            SynonymReplacement(config.synonym_replacement_prob),
        ]),
        CompositeAugmenter([
            SentenceShuffle(config.sentence_shuffle_prob * 1.5),
            WordShuffle(config.word_shuffle_window),
        ]),
        CompositeAugmenter([
            SpanMasking(config.span_mask_prob, config.span_mask_max_length),
            RandomInsertion(0.05),
        ]),
    ]
    
    return RandomAugmenter(augmenters)


if __name__ == "__main__":
    # Test augmentations
    config = AugmentationConfig()
    
    sample_text = """
    Deep Learning for Natural Language Processing. 
    We present a novel approach to understanding text using transformer architectures.
    Our method achieves state-of-the-art results on multiple benchmarks.
    The key innovation is the attention mechanism that captures long-range dependencies.
    Experiments show significant improvements over baseline methods.
    """.strip()
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    augmenters = [
        ("Word Dropout", WordDropout(0.2)),
        ("Word Shuffle", WordShuffle(3)),
        ("Sentence Shuffle", SentenceShuffle(1.0)),
        ("Synonym Replacement", SynonymReplacement(0.3)),
        ("Span Masking", SpanMasking(0.2)),
        ("Random Insertion", RandomInsertion(0.1)),
    ]
    
    for name, aug in augmenters:
        print(f"{name}:")
        print(aug(sample_text))
        print()
