import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy.interpolate import interp1d
from tqdm import tqdm


class SWaTPreprocessor:
    def __init__(self, config):
        self.config = config
        self.window_size = config['data']['window_size']
        self.stride = config['data']['stride']
        self.raw_dir = Path(config['data']['raw_dir']) / 'swat'
        
        self.normal_file = self.raw_dir / config['data']['swat']['normal_file']
        self.attack_file = self.raw_dir / config['data']['swat']['attack_file']
        self.sheet_normal = config['data']['swat']['sheet_name_normal']
        self.sheet_attack = config['data']['swat']['sheet_name_attack']
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        normal_df = pd.read_excel(self.normal_file, sheet_name=self.sheet_normal)
        attack_df = pd.read_excel(self.attack_file, sheet_name=self.sheet_attack)
        
        normal_df = normal_df.iloc[:, 1:]
        attack_df = attack_df.iloc[:, 1:]
        
        if 'Timestamp' in normal_df.columns:
            normal_df = normal_df.drop('Timestamp', axis=1)
        if 'Timestamp' in attack_df.columns:
            attack_df = attack_df.drop('Timestamp', axis=1)
        
        label_col = None
        for col in attack_df.columns:
            if 'normal' in col.lower() and 'attack' in col.lower():
                label_col = col
                break
        
        if label_col:
            attack_labels = attack_df[label_col].map({'Normal': 0, 'Attack': 1, 'A ttack': 1})
            attack_labels = attack_labels.fillna(0).astype(int)
            attack_data = attack_df.drop(label_col, axis=1)
        else:
            attack_labels = pd.Series(np.zeros(len(attack_df)), dtype=int)
            attack_data = attack_df
        
        normal_data = normal_df.select_dtypes(include=[np.number])
        attack_data = attack_data.select_dtypes(include=[np.number])
        
        normal_data = normal_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        attack_data = attack_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return normal_data, attack_data, attack_labels
    
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1.0
        normalized = (data - mean) / std
        return normalized, mean, std
    
    def create_windows(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        windows = []
        window_labels = []
        
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]
            label_window = labels[i:i + self.window_size]
            
            if len(np.unique(label_window)) == 1:
                label = label_window[0]
            else:
                label = int(np.mean(label_window) > 0.5)
            
            windows.append(window)
            window_labels.append(label)
        
        return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)
    
    def assign_severity(self, labels: np.ndarray) -> np.ndarray:
        severity = np.zeros(len(labels), dtype=np.int64)
        
        attack_indices = np.where(labels > 0)[0]
        
        if len(attack_indices) > 0:
            attack_regions = []
            start = attack_indices[0]
            
            for i in range(1, len(attack_indices)):
                if attack_indices[i] - attack_indices[i-1] > 10:
                    attack_regions.append((start, attack_indices[i-1]))
                    start = attack_indices[i]
            attack_regions.append((start, attack_indices[-1]))
            
            for start_idx, end_idx in attack_regions:
                region_len = end_idx - start_idx + 1
                
                if region_len < 50:
                    sev = 1
                elif region_len < 200:
                    sev = 2
                else:
                    sev = 3
                
                severity[start_idx:end_idx+1] = sev
        
        return severity
    
    def process(self) -> Dict[str, np.ndarray]:
        normal_df, attack_df, attack_labels = self.load_data()
        
        normal_data = normal_df.values
        attack_data = attack_df.values
        attack_labels = attack_labels.values
        
        all_data = np.vstack([normal_data, attack_data])
        normalized_data, mean, std = self.normalize(all_data)
        
        normal_normalized = normalized_data[:len(normal_data)]
        attack_normalized = normalized_data[len(normal_data):]
        
        normal_labels = np.zeros(len(normal_data), dtype=np.int64)
        normal_severity = np.zeros(len(normal_data), dtype=np.int64)
        attack_severity = self.assign_severity(attack_labels)
        
        train_windows, train_labels = self.create_windows(normal_normalized, normal_labels)
        test_windows, test_labels = self.create_windows(attack_normalized, attack_labels)
        
        train_severity_windows = []
        for i in range(0, len(normal_severity) - self.window_size + 1, self.stride):
            sev = int(np.mean(normal_severity[i:i + self.window_size]))
            train_severity_windows.append(sev)
        
        test_severity_windows = []
        for i in range(0, len(attack_severity) - self.window_size + 1, self.stride):
            sev = int(np.mean(attack_severity[i:i + self.window_size]))
            test_severity_windows.append(sev)
        
        return {
            'train_windows': train_windows,
            'train_labels': train_labels,
            'train_severity': np.array(train_severity_windows, dtype=np.int64),
            'test_windows': test_windows,
            'test_labels': test_labels,
            'test_severity': np.array(test_severity_windows, dtype=np.int64),
            'mean': mean,
            'std': std,
            'feature_names': list(normal_df.columns)
        }


class WADIPreprocessor:
    def __init__(self, config):
        self.config = config
        self.window_size = config['data']['window_size']
        self.stride = config['data']['stride']
        self.raw_dir = Path(config['data']['raw_dir']) / 'wadi'
        
        self.train_file = self.raw_dir / config['data']['wadi']['train_file']
        self.attack_file = self.raw_dir / config['data']['wadi']['attack_file']
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        train_df = pd.read_csv(self.train_file)
        attack_df = pd.read_csv(self.attack_file)
        
        if 'Row' in train_df.columns:
            train_df = train_df.drop('Row', axis=1)
        if 'Row' in attack_df.columns:
            attack_df = attack_df.drop('Row', axis=1)
        
        for col in ['Date', 'Time', 'Timestamp']:
            if col in train_df.columns:
                train_df = train_df.drop(col, axis=1)
            if col in attack_df.columns:
                attack_df = attack_df.drop(col, axis=1)
        
        label_col = None
        for col in attack_df.columns:
            if 'attack' in col.lower() or 'label' in col.lower():
                label_col = col
                break
        
        if label_col:
            attack_labels = attack_df[label_col].map({-1: 1, 1: 0})
            attack_labels = attack_labels.fillna(0).astype(int)
            attack_data = attack_df.drop(label_col, axis=1)
        else:
            attack_labels = pd.Series(np.zeros(len(attack_df)), dtype=int)
            attack_data = attack_df
        
        train_data = train_df.select_dtypes(include=[np.number])
        attack_data = attack_data.select_dtypes(include=[np.number])
        
        train_data = train_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        attack_data = attack_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return train_data, attack_data, attack_labels
    
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1.0
        normalized = (data - mean) / std
        return normalized, mean, std
    
    def create_windows(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        windows = []
        window_labels = []
        
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]
            label_window = labels[i:i + self.window_size]
            
            if len(np.unique(label_window)) == 1:
                label = label_window[0]
            else:
                label = int(np.mean(label_window) > 0.5)
            
            windows.append(window)
            window_labels.append(label)
        
        return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)
    
    def assign_severity(self, labels: np.ndarray) -> np.ndarray:
        severity = np.zeros(len(labels), dtype=np.int64)
        
        attack_indices = np.where(labels > 0)[0]
        
        if len(attack_indices) > 0:
            attack_regions = []
            start = attack_indices[0]
            
            for i in range(1, len(attack_indices)):
                if attack_indices[i] - attack_indices[i-1] > 10:
                    attack_regions.append((start, attack_indices[i-1]))
                    start = attack_indices[i]
            attack_regions.append((start, attack_indices[-1]))
            
            for start_idx, end_idx in attack_regions:
                region_len = end_idx - start_idx + 1
                
                if region_len < 50:
                    sev = 1
                elif region_len < 200:
                    sev = 2
                else:
                    sev = 3
                
                severity[start_idx:end_idx+1] = sev
        
        return severity
    
    def process(self) -> Dict[str, np.ndarray]:
        train_df, attack_df, attack_labels = self.load_data()
        
        train_data = train_df.values
        attack_data = attack_df.values
        attack_labels = attack_labels.values
        
        all_data = np.vstack([train_data, attack_data])
        normalized_data, mean, std = self.normalize(all_data)
        
        train_normalized = normalized_data[:len(train_data)]
        attack_normalized = normalized_data[len(train_data):]
        
        train_labels = np.zeros(len(train_data), dtype=np.int64)
        train_severity = np.zeros(len(train_data), dtype=np.int64)
        attack_severity = self.assign_severity(attack_labels)
        
        train_windows, train_window_labels = self.create_windows(train_normalized, train_labels)
        attack_windows, attack_window_labels = self.create_windows(attack_normalized, attack_labels)
        
        train_severity_windows = []
        for i in range(0, len(train_severity) - self.window_size + 1, self.stride):
            sev = int(np.mean(train_severity[i:i + self.window_size]))
            train_severity_windows.append(sev)
        
        attack_severity_windows = []
        for i in range(0, len(attack_severity) - self.window_size + 1, self.stride):
            sev = int(np.mean(attack_severity[i:i + self.window_size]))
            attack_severity_windows.append(sev)
        
        all_windows = np.vstack([train_windows, attack_windows])
        all_labels = np.concatenate([train_window_labels, attack_window_labels])
        all_severity = np.concatenate([train_severity_windows, attack_severity_windows])
        
        return {
            'windows': all_windows,
            'labels': all_labels,
            'severity': all_severity,
            'mean': mean,
            'std': std,
            'feature_names': list(train_df.columns)
        }


class WESADPreprocessor:
    def __init__(self, config):
        self.config = config
        self.window_size = config['data']['window_size']
        self.stride = config['data']['stride']
        self.target_fs = config['data']['wesad']['target_fs']
        self.subjects = config['data']['wesad']['subjects']
        self.raw_dir = Path(config['data']['raw_dir']) / 'wesad'
    
    def load_subject(self, subject_id: int) -> Dict:
        subject_path = self.raw_dir / f'S{subject_id}' / f'S{subject_id}.pkl'
        
        with open(subject_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        return data
    
    def resample_signal(self, signal: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
        if len(signal) == 0:
            return np.array([])
        
        duration = len(signal) / original_fs
        target_length = int(duration * target_fs)
        
        if target_length == 0:
            return np.array([])
        
        time_original = np.linspace(0, duration, len(signal))
        time_target = np.linspace(0, duration, target_length)
        
        f = interp1d(time_original, signal, kind='linear', fill_value='extrapolate')
        resampled = f(time_target)
        
        return resampled
    
    def process_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        data = self.load_subject(subject_id)
        
        chest_fs = self.config['data']['wesad']['chest_fs']
        
        chest = data['signal']['chest']
        labels = data['label'].flatten()
        
        valid_labels = np.isin(labels, [1, 2, 3, 4])
        
        ecg = chest['ECG'].flatten()[valid_labels]
        eda = chest['EDA'].flatten()[valid_labels]
        emg = chest['EMG'].flatten()[valid_labels]
        resp = chest['Resp'].flatten()[valid_labels]
        temp = chest['Temp'].flatten()[valid_labels]
        acc = chest['ACC'][valid_labels]
        
        labels_filtered = labels[valid_labels]
        
        ecg_r = self.resample_signal(ecg, chest_fs, self.target_fs)
        eda_r = self.resample_signal(eda, chest_fs, self.target_fs)
        emg_r = self.resample_signal(emg, chest_fs, self.target_fs)
        resp_r = self.resample_signal(resp, chest_fs, self.target_fs)
        temp_r = self.resample_signal(temp, chest_fs, self.target_fs)
        
        acc_x_r = self.resample_signal(acc[:, 0], chest_fs, self.target_fs)
        acc_y_r = self.resample_signal(acc[:, 1], chest_fs, self.target_fs)
        acc_z_r = self.resample_signal(acc[:, 2], chest_fs, self.target_fs)
        
        labels_r = self.resample_signal(labels_filtered.astype(float), chest_fs, self.target_fs)
        
        min_length = min(len(ecg_r), len(eda_r), len(emg_r), len(resp_r), 
                        len(temp_r), len(acc_x_r), len(acc_y_r), len(acc_z_r), len(labels_r))
        
        features = np.column_stack([
            ecg_r[:min_length],
            eda_r[:min_length],
            emg_r[:min_length],
            resp_r[:min_length],
            temp_r[:min_length],
            acc_x_r[:min_length],
            acc_y_r[:min_length],
            acc_z_r[:min_length]
        ])
        
        labels_final = labels_r[:min_length]
        
        return features, labels_final
    
    def create_windows(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        window_samples = int(self.window_size * self.target_fs)
        stride_samples = int(self.stride * self.target_fs)
        
        windows = []
        window_labels = []
        
        for i in range(0, len(features) - window_samples + 1, stride_samples):
            window = features[i:i + window_samples]
            label_window = labels[i:i + window_samples]
            
            label_counts = np.bincount(label_window.astype(int))
            if len(label_counts) > 0:
                label = int(np.argmax(label_counts))
            else:
                label = 0
            
            if label in [1, 2, 3]:
                windows.append(window)
                window_labels.append(label)
        
        return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)
    
    def map_stress_labels(self, labels: np.ndarray) -> np.ndarray:
        stress_mapping = {1: 0, 2: 1, 3: 0, 4: 0}
        return np.array([stress_mapping.get(int(l), 0) for l in labels], dtype=np.int64)
    
    def assign_stress_levels(self, labels: np.ndarray) -> np.ndarray:
        stress = np.zeros(len(labels), dtype=np.int64)
        
        for i, label in enumerate(labels):
            if label == 0:
                stress[i] = 0
            elif label == 1:
                stress[i] = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            else:
                stress[i] = 0
        
        return stress
    
    def process(self) -> Dict[str, np.ndarray]:
        all_windows = []
        all_labels = []
        all_stress = []
        
        for subject_id in tqdm(self.subjects, desc='Processing WESAD'):
            try:
                features, labels = self.process_subject(subject_id)
                windows, window_labels = self.create_windows(features, labels)
                
                if len(windows) > 0:
                    stress_binary = self.map_stress_labels(window_labels)
                    stress_levels = self.assign_stress_levels(stress_binary)
                    
                    all_windows.append(windows)
                    all_labels.append(stress_binary)
                    all_stress.append(stress_levels)
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                continue
        
        if len(all_windows) == 0:
            return {
                'windows': np.array([], dtype=np.float32).reshape(0, 0, 8),
                'labels': np.array([], dtype=np.int64),
                'stress': np.array([], dtype=np.int64),
                'mean': np.zeros(8),
                'std': np.ones(8)
            }
        
        all_windows = np.vstack(all_windows)
        all_labels = np.concatenate(all_labels)
        all_stress = np.concatenate(all_stress)
        
        mean = all_windows.reshape(-1, all_windows.shape[-1]).mean(axis=0)
        std = all_windows.reshape(-1, all_windows.shape[-1]).std(axis=0)
        std[std == 0] = 1.0
        
        all_windows_normalized = (all_windows - mean) / std
        
        return {
            'windows': all_windows_normalized,
            'labels': all_labels,
            'stress': all_stress,
            'mean': mean,
            'std': std
        }


class SWELLPreprocessor:
    def __init__(self, config):
        self.config = config
        self.window_size = config['data']['window_size']
        self.stride = config['data']['stride']
        self.raw_dir = Path(config['data']['raw_dir']) / 'swell'
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        all_features = []
        all_labels = []
        
        for csv_file in self.raw_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                
                label_col = None
                for col in df.columns:
                    if 'label' in col.lower() or 'workload' in col.lower():
                        label_col = col
                        break
                
                if label_col:
                    labels = df[label_col].values
                    features = df.drop(label_col, axis=1).select_dtypes(include=[np.number]).values
                else:
                    labels = np.zeros(len(df))
                    features = df.select_dtypes(include=[np.number]).values
                
                all_features.append(features)
                all_labels.append(labels)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        if len(all_features) == 0:
            return np.array([]), np.array([])
        
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        
        return all_features, all_labels
    
    def create_windows(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        windows = []
        window_labels = []
        
        for i in range(0, len(features) - self.window_size + 1, self.stride):
            window = features[i:i + self.window_size]
            label = np.mean(labels[i:i + self.window_size])
            
            windows.append(window)
            window_labels.append(label)
        
        return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.float32)
    
    def assign_workload_levels(self, labels: np.ndarray) -> np.ndarray:
        workload = np.zeros(len(labels), dtype=np.int64)
        
        for i, label in enumerate(labels):
            if label < 0.33:
                workload[i] = 0
            elif label < 0.67:
                workload[i] = 1
            else:
                workload[i] = 2
        
        return workload
    
    def process(self) -> Dict[str, np.ndarray]:
        features, labels = self.load_data()
        
        if len(features) == 0:
            return {
                'windows': np.array([], dtype=np.float32).reshape(0, 0, 0),
                'labels': np.array([], dtype=np.float32),
                'workload': np.array([], dtype=np.int64),
                'mean': np.array([]),
                'std': np.array([])
            }
        
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        
        features_normalized = (features - mean) / std
        
        windows, window_labels = self.create_windows(features_normalized, labels)
        workload = self.assign_workload_levels(window_labels)
        
        return {
            'windows': windows,
            'labels': window_labels,
            'workload': workload,
            'mean': mean,
            'std': std
        }