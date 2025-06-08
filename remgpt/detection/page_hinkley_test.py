"""
Page-Hinkley test for detecting changes in data streams.
"""


class PageHinkleyTest:
    """
    Page-Hinkley test for detecting changes in data streams.
    
    This implementation detects drift in the cosine similarity between
    consecutive message embeddings.
    """
    
    def __init__(self, threshold: float = 0.5, alpha: float = 0.05):
        """
        Initialize Page-Hinkley test.
        
        Args:
            threshold: Detection threshold for drift
            alpha: Significance level for the test
        """
        self.threshold = threshold
        self.alpha = alpha
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.n_samples = 0
        self.mean_estimate = 0.0
        self.baseline_similarity = 0.8  # Expected similarity for same topic
        
    def reset(self):
        """Reset the test statistics."""
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.n_samples = 0
        self.mean_estimate = 0.0
    
    def add_sample(self, similarity: float) -> bool:
        """
        Add a new similarity sample and check for drift.
        
        Args:
            similarity: Cosine similarity between current and previous message
            
        Returns:
            True if drift is detected, False otherwise
        """
        # Update mean estimate
        self.n_samples += 1
        self.mean_estimate += (similarity - self.mean_estimate) / self.n_samples
        
        # Update cumulative sum using baseline instead of running mean
        # We expect high similarity (close to baseline) for same topic
        deviation = self.baseline_similarity - similarity
        self.cumulative_sum += deviation
        
        # Update minimum cumulative sum
        if self.cumulative_sum < self.min_cumulative_sum:
            self.min_cumulative_sum = self.cumulative_sum
        
        # Test statistic
        test_statistic = self.cumulative_sum - self.min_cumulative_sum
        
        # Drift detected if test statistic exceeds threshold
        return test_statistic > self.threshold 