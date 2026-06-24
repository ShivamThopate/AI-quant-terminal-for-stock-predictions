"""
FinBERT Sentiment Scorer
=========================
Uses the ProsusAI/finbert transformer model to score financial text
with domain-specific understanding.

Unlike VADER (dictionary-based word counting), FinBERT is a deep learning
model pre-trained on financial news and earnings calls. It understands
context, e.g. "profit fell less than expected" is actually positive.

The model is loaded lazily and cached in memory so it only downloads
once and loads once per session.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Lazy-loaded singleton to avoid downloading on import
_pipeline = None
_load_failed = False


def _get_pipeline():
    """
    Lazy-load the FinBERT sentiment pipeline.

    Downloads the model on first run (~420MB), then caches it
    locally in the HuggingFace cache directory.
    """
    global _pipeline, _load_failed

    if _pipeline is not None:
        return _pipeline

    if _load_failed:
        return None

    try:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading FinBERT model (first run downloads ~420MB)...")
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU — use 0 for GPU if available
            top_k=None,  # Return all 3 class probabilities
        )
        logger.info("FinBERT model loaded successfully.")
        return _pipeline

    except ImportError:
        logger.error(
            "transformers or torch not installed. "
            "Run: pip install transformers torch"
        )
        _load_failed = True
        return None

    except Exception as e:
        logger.error("Failed to load FinBERT model: %s", e)
        _load_failed = True
        return None


class FinBERTScorer:
    """
    Score financial text using the ProsusAI/finbert transformer model.

    Returns a compound score in [-1.0, +1.0] matching the same interface
    as the old VADER scorer, so it's a drop-in replacement.

    Scoring logic:
        score = P(positive) - P(negative)

    This gives:
        - Headlines with strong positive sentiment → close to +1.0
        - Headlines with strong negative sentiment → close to -1.0
        - Neutral/mixed headlines → close to 0.0
    """

    def score(self, text: str) -> float:
        """
        Score a single text string.

        Returns a float in [-1.0, +1.0].
        Falls back to 0.0 (neutral) if FinBERT fails to load.
        """
        pipe = _get_pipeline()
        if pipe is None:
            # Fallback: return neutral if model can't load
            return self._vader_fallback(text)

        try:
            # FinBERT returns: [{'label': 'positive', 'score': 0.85}, ...]
            # With top_k=None, we get all 3 classes
            results = pipe(text[:512])  # BERT max token limit

            # results is a list of lists when top_k=None
            scores_list = results[0] if results else []

            p_positive = 0.0
            p_negative = 0.0

            for item in scores_list:
                label = item["label"].lower()
                if label == "positive":
                    p_positive = item["score"]
                elif label == "negative":
                    p_negative = item["score"]

            # Compound score: same range as VADER [-1, +1]
            return round(p_positive - p_negative, 4)

        except Exception as e:
            logger.warning("FinBERT scoring failed for text: %s — %s", text[:50], e)
            return self._vader_fallback(text)

    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Score multiple texts efficiently using batch inference.

        FinBERT processes batches faster than scoring one-by-one
        because the GPU/CPU can parallelize the computation.
        """
        pipe = _get_pipeline()
        if pipe is None:
            return [self._vader_fallback(t) for t in texts]

        try:
            # Truncate to BERT's max length
            truncated = [t[:512] for t in texts]
            all_results = pipe(truncated, batch_size=16)

            scores = []
            for result in all_results:
                p_positive = 0.0
                p_negative = 0.0
                for item in result:
                    label = item["label"].lower()
                    if label == "positive":
                        p_positive = item["score"]
                    elif label == "negative":
                        p_negative = item["score"]
                scores.append(round(p_positive - p_negative, 4))

            return scores

        except Exception as e:
            logger.warning("FinBERT batch scoring failed: %s", e)
            return [self._vader_fallback(t) for t in texts]

    @staticmethod
    def _vader_fallback(text: str) -> float:
        """
        Emergency fallback to VADER if FinBERT can't load.

        This ensures the system never fully breaks — it just
        degrades gracefully to the old scoring method.
        """
        try:
            import nltk
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                from nltk.sentiment.vader import SentimentIntensityAnalyzer

            vader = SentimentIntensityAnalyzer()
            return vader.polarity_scores(text)["compound"]

        except Exception:
            return 0.0  # Last resort: neutral
