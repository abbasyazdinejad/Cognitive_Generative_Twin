import numpy as np
from typing import Optional


class DataAugmenter:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def add_gaussian_noise(
        self,
        data: np.ndarray,
        noise_level: float,
        fraction_features: float
    ) -> np.ndarray:
        
        noisy_data = data.copy()
        
        num_features = data.shape[-1]
        num_features_to_corrupt = int(num_features * fraction_features)
        
        if num_features_to_corrupt == 0:
            return noisy_data
        
        for i in range(len(noisy_data)):
            features_to_corrupt = self.rng.choice(
                num_features,
                num_features_to_corrupt,
                replace=False
            )
            
            for feature_idx in features_to_corrupt:
                feature_std = np.std(data[:, :, feature_idx])
                noise = self.rng.normal(0, noise_level * feature_std, size=data.shape[1])
                noisy_data[i, :, feature_idx] += noise
        
        return noisy_data
    
    def apply_dropout(
        self,
        data: np.ndarray,
        dropout_rate: float
    ) -> np.ndarray:
        
        dropped_data = data.copy()
        
        num_channels = data.shape[-1]
        num_channels_to_drop = int(num_channels * dropout_rate)
        
        if num_channels_to_drop == 0:
            return dropped_data
        
        channels_to_drop = self.rng.choice(
            num_channels,
            num_channels_to_drop,
            replace=False
        )
        
        dropped_data[:, :, channels_to_drop] = 0
        
        return dropped_data
    
    def time_warp(
        self,
        data: np.ndarray,
        warp_factor: float = 0.2
    ) -> np.ndarray:
        
        warped_data = []
        
        for window in data:
            original_length = window.shape[0]
            new_length = int(original_length * (1 + self.rng.uniform(-warp_factor, warp_factor)))
            new_length = max(10, new_length)
            
            indices = np.linspace(0, original_length - 1, new_length)
            
            warped_window = np.zeros((new_length, window.shape[1]))
            
            for feature_idx in range(window.shape[1]):
                warped_window[:, feature_idx] = np.interp(
                    indices,
                    np.arange(original_length),
                    window[:, feature_idx]
                )
            
            if new_length > original_length:
                warped_window = warped_window[:original_length]
            elif new_length < original_length:
                pad_length = original_length - new_length
                warped_window = np.pad(
                    warped_window,
                    ((0, pad_length), (0, 0)),
                    mode='edge'
                )
            
            warped_data.append(warped_window)
        
        return np.array(warped_data, dtype=data.dtype)
    
    def magnitude_scaling(
        self,
        data: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        
        scaled_data = []
        
        for window in data:
            scale = self.rng.uniform(scale_range[0], scale_range[1])
            scaled_window = window * scale
            scaled_data.append(scaled_window)
        
        return np.array(scaled_data, dtype=data.dtype)
    
    def time_shift(
        self,
        data: np.ndarray,
        max_shift: int = 10
    ) -> np.ndarray:
        
        shifted_data = []
        
        for window in data:
            shift = self.rng.integers(-max_shift, max_shift + 1)
            
            if shift > 0:
                shifted_window = np.pad(window, ((shift, 0), (0, 0)), mode='edge')[:len(window)]
            elif shift < 0:
                shifted_window = np.pad(window, ((0, -shift), (0, 0)), mode='edge')[-len(window):]
            else:
                shifted_window = window
            
            shifted_data.append(shifted_window)
        
        return np.array(shifted_data, dtype=data.dtype)
    
    def augment_batch(
        self,
        data: np.ndarray,
        augmentation_config: Dict[str, any]
    ) -> np.ndarray:
        
        augmented = data.copy()
        
        if augmentation_config.get('gaussian_noise', False):
            noise_level = augmentation_config.get('noise_level', 0.1)
            fraction = augmentation_config.get('noise_fraction', 0.1)
            augmented = self.add_gaussian_noise(augmented, noise_level, fraction)
        
        if augmentation_config.get('dropout', False):
            dropout_rate = augmentation_config.get('dropout_rate', 0.1)
            augmented = self.apply_dropout(augmented, dropout_rate)
        
        if augmentation_config.get('time_warp', False):
            warp_factor = augmentation_config.get('warp_factor', 0.2)
            augmented = self.time_warp(augmented, warp_factor)
        
        if augmentation_config.get('magnitude_scaling', False):
            scale_range = augmentation_config.get('scale_range', (0.8, 1.2))
            augmented = self.magnitude_scaling(augmented, scale_range)
        
        if augmentation_config.get('time_shift', False):
            max_shift = augmentation_config.get('max_shift', 10)
            augmented = self.time_shift(augmented, max_shift)
        
        return augmented