"""Perceptual hash computation from DCT."""

import numpy as np
from typing import Dict, Any, List

from ..base import BaseFeature


class PerceptualHashes(BaseFeature):
    """Compute perceptual hashes for frame similarity."""
    
    FEATURE_NAME = 'perceptual_hashes'
    REQUIRED_ANALYSES = ['dct_transform']
    
    def __init__(self, hamming_threshold: int = 10):
        """
        Args:
            hamming_threshold: Threshold for considering frames similar
        """
        super().__init__()
        self.hamming_threshold = hamming_threshold
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute perceptual hash similarities.
        
        Returns:
            Dict with hash similarities and duplicate frames
        """
        self.validate_inputs(analysis_data)
        
        dct_analysis = analysis_data['dct_transform']
        hashes = dct_analysis.data.get('perceptual_hashes')
        
        if hashes is None or len(hashes) == 0:
            return {
                'similarities': [],
                'duplicate_frames': [],
                'unique_frames': 0
            }
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(hashes) - 1):
            distance = self._hamming_distance(hashes[i], hashes[i+1])
            similarity = 1.0 - (distance / (len(hashes[i]) * 8))  # Convert to similarity
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Find duplicate/similar frames
        duplicate_groups = self._find_duplicates(hashes)
        
        # Compute unique frame count
        unique_frames = len(hashes) - sum(len(g) - 1 for g in duplicate_groups)
        
        return {
            'similarities': similarities,
            'mean_similarity': float(np.mean(similarities)) if len(similarities) > 0 else 0,
            'duplicate_groups': duplicate_groups,
            'unique_frames': unique_frames,
            'similarity_matrix': self._compute_similarity_matrix(hashes)
        }
    
    def _hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> int:
        """Compute Hamming distance between two hashes."""
        
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2)) * 8  # Maximum distance
        
        distance = 0
        for b1, b2 in zip(hash1, hash2):
            # XOR and count set bits
            xor = b1 ^ b2
            distance += bin(xor).count('1')
        
        return distance
    
    def _find_duplicates(self, hashes: np.ndarray) -> List[List[int]]:
        """Find groups of duplicate/similar frames."""
        n_frames = len(hashes)
        visited = set()
        duplicate_groups = []
        
        for i in range(n_frames):
            if i in visited:
                continue
            
            group = [i]
            visited.add(i)
            
            for j in range(i + 1, n_frames):
                if j not in visited:
                    distance = self._hamming_distance(hashes[i], hashes[j])
                    if distance <= self.hamming_threshold:
                        group.append(j)
                        visited.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    def _compute_similarity_matrix(self, hashes: np.ndarray) -> np.ndarray:
        """Compute full similarity matrix (for small videos)."""
        n_frames = len(hashes)
        
        # Only compute for small videos to avoid memory issues
        if n_frames > 1000:
            return np.array([])
        
        similarity_matrix = np.zeros((n_frames, n_frames))
        
        for i in range(n_frames):
            for j in range(i, n_frames):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    distance = self._hamming_distance(hashes[i], hashes[j])
                    similarity = 1.0 - (distance / (len(hashes[i]) * 8))
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
