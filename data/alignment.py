import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from typing import Dict, List, Tuple


class CrossDomainAligner:
    def __init__(self, config):
        self.config = config
        self.seed = config['project']['seed']
        self.rng = np.random.default_rng(self.seed)
    
    def compute_severity_stress_correlation(
        self,
        severity: np.ndarray,
        stress: np.ndarray
    ) -> Tuple[float, float]:
        
        valid_mask = (severity >= 0) & (stress >= 0)
        severity_valid = severity[valid_mask]
        stress_valid = stress[valid_mask]
        
        if len(severity_valid) < 10:
            return 0.0, 1.0
        
        correlation, p_value = spearmanr(severity_valid, stress_valid)
        
        return float(correlation), float(p_value)
    
    def create_alignment_mapping(
        self,
        cps_severity: np.ndarray,
        bio_stress: np.ndarray
    ) -> Dict[int, int]:
        
        mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3
        }
        
        return mapping
    
    def align_distributions(
        self,
        cps_data: Dict,
        bio_data: Dict
    ) -> List[Tuple[int, int]]:
        
        cps_severity = cps_data['severity']
        bio_stress = bio_data['stress']
        
        severity_bins = np.digitize(cps_severity, bins=[0, 1, 2, 3])
        stress_bins = np.digitize(bio_stress, bins=[0, 1, 2, 3])
        
        aligned_pairs = []
        
        for sev_bin in range(1, 5):
            cps_indices = np.where(severity_bins == sev_bin)[0]
            bio_indices = np.where(stress_bins == sev_bin)[0]
            
            min_len = min(len(cps_indices), len(bio_indices))
            
            if min_len > 0:
                cps_sample = self.rng.choice(cps_indices, min_len, replace=False)
                bio_sample = self.rng.choice(bio_indices, min_len, replace=False)
                
                for c, b in zip(cps_sample, bio_sample):
                    aligned_pairs.append((int(c), int(b)))
        
        return aligned_pairs
    
    def sample_conditional_pairs(
        self,
        cps_data: Dict,
        bio_data: Dict,
        beh_data: Dict,
        num_samples: int
    ) -> List[Dict]:
        
        pairs = []
        
        cps_severity = cps_data['severity']
        bio_stress = bio_data['stress']
        beh_workload = beh_data['workload']
        
        severity_levels = np.unique(cps_severity)
        stress_levels = np.unique(bio_stress)
        
        all_levels = sorted(set(severity_levels) | set(stress_levels))
        
        samples_per_level = max(1, num_samples // len(all_levels))
        
        for level in all_levels:
            cps_candidates = np.where(cps_severity == level)[0]
            bio_candidates = np.where(bio_stress == level)[0]
            beh_candidates = np.where(beh_workload <= level)[0]
            
            if len(beh_candidates) == 0:
                beh_candidates = np.arange(len(beh_workload))
            
            n_samples = min(
                samples_per_level,
                len(cps_candidates),
                len(bio_candidates),
                len(beh_candidates)
            )
            
            if n_samples > 0:
                cps_sample = self.rng.choice(cps_candidates, n_samples, replace=False)
                bio_sample = self.rng.choice(bio_candidates, n_samples, replace=False)
                beh_sample = self.rng.choice(beh_candidates, n_samples, replace=False)
                
                for c, b, beh in zip(cps_sample, bio_sample, beh_sample):
                    pairs.append({
                        'cps_idx': int(c),
                        'bio_idx': int(b),
                        'beh_idx': int(beh),
                        'severity': int(level),
                        'stress': int(level)
                    })
        
        if len(pairs) < num_samples:
            while len(pairs) < num_samples:
                severity_level = self.rng.choice(all_levels)
                
                cps_candidates = np.where(cps_severity == severity_level)[0]
                bio_candidates = np.where(bio_stress == severity_level)[0]
                beh_candidates = np.arange(len(beh_workload))
                
                if len(cps_candidates) > 0 and len(bio_candidates) > 0:
                    cps_idx = self.rng.choice(cps_candidates)
                    bio_idx = self.rng.choice(bio_candidates)
                    beh_idx = self.rng.choice(beh_candidates)
                    
                    pairs.append({
                        'cps_idx': int(cps_idx),
                        'bio_idx': int(bio_idx),
                        'beh_idx': int(beh_idx),
                        'severity': int(severity_level),
                        'stress': int(severity_level)
                    })
        
        self.rng.shuffle(pairs)
        
        return pairs[:num_samples]
    
    def compute_distribution_distance(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray
    ) -> float:
        
        hist1, _ = np.histogram(dist1, bins=4, range=(0, 4), density=True)
        hist2, _ = np.histogram(dist2, bins=4, range=(0, 4), density=True)
        
        distance = wasserstein_distance(hist1, hist2)
        
        return float(distance)
    
    def validate_alignment(
        self,
        cps_data: Dict,
        bio_data: Dict,
        aligned_pairs: List[Dict]
    ) -> Dict[str, float]:
        
        if len(aligned_pairs) == 0:
            return {
                'correlation': 0.0,
                'p_value': 1.0,
                'wasserstein_dist': 1.0,
                'alignment_quality': 0.0
            }
        
        cps_severity = cps_data['severity']
        bio_stress = bio_data['stress']
        
        aligned_severities = [cps_severity[p['cps_idx']] for p in aligned_pairs]
        aligned_stresses = [bio_stress[p['bio_idx']] for p in aligned_pairs]
        
        correlation, p_value = self.compute_severity_stress_correlation(
            np.array(aligned_severities),
            np.array(aligned_stresses)
        )
        
        w_dist = self.compute_distribution_distance(
            np.array(aligned_severities),
            np.array(aligned_stresses)
        )
        
        exact_matches = sum(1 for s, st in zip(aligned_severities, aligned_stresses) if s == st)
        alignment_quality = exact_matches / len(aligned_pairs)
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'wasserstein_dist': w_dist,
            'alignment_quality': alignment_quality
        }


class ConditionalSampler:
    def __init__(self, config):
        self.config = config
        self.seed = config['project']['seed']
        self.rng = np.random.default_rng(self.seed)
    
    def sample_by_condition(
        self,
        data: Dict,
        condition_key: str,
        condition_value: int,
        num_samples: int
    ) -> np.ndarray:
        
        condition_array = data[condition_key]
        candidates = np.where(condition_array == condition_value)[0]
        
        if len(candidates) == 0:
            return np.array([], dtype=np.int64)
        
        n_samples = min(num_samples, len(candidates))
        
        sampled_indices = self.rng.choice(candidates, n_samples, replace=False)
        
        return sampled_indices
    
    def stratified_sample(
        self,
        data: Dict,
        condition_key: str,
        num_samples_per_class: int
    ) -> Dict[int, np.ndarray]:
        
        condition_array = data[condition_key]
        unique_values = np.unique(condition_array)
        
        stratified_samples = {}
        
        for value in unique_values:
            indices = self.sample_by_condition(
                data,
                condition_key,
                int(value),
                num_samples_per_class
            )
            stratified_samples[int(value)] = indices
        
        return stratified_samples
    
    def balanced_sample(
        self,
        data: Dict,
        condition_key: str,
        total_samples: int
    ) -> np.ndarray:
        
        condition_array = data[condition_key]
        unique_values = np.unique(condition_array)
        
        samples_per_class = total_samples // len(unique_values)
        
        all_indices = []
        
        for value in unique_values:
            indices = self.sample_by_condition(
                data,
                condition_key,
                int(value),
                samples_per_class
            )
            all_indices.extend(indices)
        
        all_indices = np.array(all_indices, dtype=np.int64)
        self.rng.shuffle(all_indices)
        
        return all_indices[:total_samples]