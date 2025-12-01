import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional


class CPSDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray, severity: np.ndarray):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
        self.severity = torch.LongTensor(severity)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'data': self.windows[idx],
            'label': self.labels[idx],
            'severity': self.severity[idx]
        }


class PhysiologyDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray, stress: np.ndarray):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
        self.stress = torch.LongTensor(stress)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'data': self.windows[idx],
            'label': self.labels[idx],
            'stress': self.stress[idx]
        }


class BehaviorDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray, workload: np.ndarray):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.FloatTensor(labels)
        self.workload = torch.LongTensor(workload)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'data': self.windows[idx],
            'label': self.labels[idx],
            'workload': self.workload[idx]
        }


class CrossDomainDataset(Dataset):
    def __init__(self, cps_data: Dict, bio_data: Dict, beh_data: Dict):
        self.cps_windows = torch.FloatTensor(cps_data['windows'])
        self.cps_severity = torch.LongTensor(cps_data['severity'])
        
        self.bio_windows = torch.FloatTensor(bio_data['windows'])
        self.bio_stress = torch.LongTensor(bio_data['stress'])
        
        self.beh_windows = torch.FloatTensor(beh_data['windows'])
        self.beh_workload = torch.LongTensor(beh_data['workload'])
        
        self.cps_len = len(self.cps_windows)
        self.bio_len = len(self.bio_windows)
        self.beh_len = len(self.beh_windows)
        
        self.max_len = max(self.cps_len, self.bio_len, self.beh_len)
    
    def __len__(self):
        return self.max_len
    
    def __getitem__(self, idx):
        cps_idx = idx % self.cps_len
        bio_idx = idx % self.bio_len
        beh_idx = idx % self.beh_len
        
        return {
            'cps': self.cps_windows[cps_idx],
            'bio': self.bio_windows[bio_idx],
            'beh': self.beh_windows[beh_idx],
            'severity': self.cps_severity[cps_idx],
            'stress': self.bio_stress[bio_idx]
        }


class AlignedCrossDomainDataset(Dataset):
    def __init__(self, aligned_pairs: list, cps_data: Dict, bio_data: Dict, beh_data: Dict):
        self.pairs = aligned_pairs
        
        self.cps_windows = torch.FloatTensor(cps_data['windows'])
        self.cps_severity = torch.LongTensor(cps_data['severity'])
        
        self.bio_windows = torch.FloatTensor(bio_data['windows'])
        self.bio_stress = torch.LongTensor(bio_data['stress'])
        
        self.beh_windows = torch.FloatTensor(beh_data['windows'])
        self.beh_workload = torch.LongTensor(beh_data['workload'])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        cps_idx = pair['cps_idx']
        bio_idx = pair['bio_idx']
        beh_idx = pair['beh_idx']
        
        return {
            'cps': self.cps_windows[cps_idx],
            'bio': self.bio_windows[bio_idx],
            'beh': self.beh_windows[beh_idx],
            'severity': self.cps_severity[cps_idx],
            'stress': self.bio_stress[bio_idx]
        }


def align_feature_dimensions(data_dict: Dict, target_dim: int) -> Dict:
    windows = data_dict['windows']
    
    if windows.size == 0:
        return data_dict
    
    current_dim = windows.shape[-1]
    
    if current_dim == target_dim:
        return data_dict
    
    if current_dim > target_dim:
        windows = windows[..., :target_dim]
    else:
        pad_width = ((0, 0), (0, 0), (0, target_dim - current_dim))
        windows = np.pad(windows, pad_width, mode='constant', constant_values=0)
    
    data_dict['windows'] = windows
    return data_dict


def create_dataloaders(
    cps_processed: Dict,
    bio_processed: Dict,
    beh_processed: Dict,
    config: Dict,
    use_alignment: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_ratio = config['data']['split']['train']
    val_ratio = config['data']['split']['val']
    seed = config['project']['seed']
    batch_size = config['training']['generative']['batch_size']
    
    target_dim = cps_processed['windows'].shape[-1]
    
    bio_processed = align_feature_dimensions(bio_processed, target_dim)
    beh_processed = align_feature_dimensions(beh_processed, target_dim)
    
    def split_data(data_dict):
        n_samples = len(data_dict['windows'])
        indices = np.arange(n_samples)
        
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )
        
        val_size = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=seed,
            shuffle=True
        )
        
        train_data = {}
        val_data = {}
        test_data = {}
        
        for key in data_dict.keys():
            if isinstance(data_dict[key], np.ndarray) and len(data_dict[key]) == n_samples:
                train_data[key] = data_dict[key][train_idx]
                val_data[key] = data_dict[key][val_idx]
                test_data[key] = data_dict[key][test_idx]
            else:
                train_data[key] = data_dict[key]
                val_data[key] = data_dict[key]
                test_data[key] = data_dict[key]
        
        return train_data, val_data, test_data
    
    cps_train, cps_val, cps_test = split_data(cps_processed)
    bio_train, bio_val, bio_test = split_data(bio_processed)
    beh_train, beh_val, beh_test = split_data(beh_processed)
    
    if use_alignment:
        from .alignment import CrossDomainAligner
        
        aligner = CrossDomainAligner(config)
        
        train_pairs = aligner.sample_conditional_pairs(
            cps_train, bio_train, beh_train,
            num_samples=min(len(cps_train['windows']), len(bio_train['windows']))
        )
        
        val_pairs = aligner.sample_conditional_pairs(
            cps_val, bio_val, beh_val,
            num_samples=min(len(cps_val['windows']), len(bio_val['windows']))
        )
        
        test_pairs = aligner.sample_conditional_pairs(
            cps_test, bio_test, beh_test,
            num_samples=min(len(cps_test['windows']), len(bio_test['windows']))
        )
        
        train_dataset = AlignedCrossDomainDataset(train_pairs, cps_train, bio_train, beh_train)
        val_dataset = AlignedCrossDomainDataset(val_pairs, cps_val, bio_val, beh_val)
        test_dataset = AlignedCrossDomainDataset(test_pairs, cps_test, bio_test, beh_test)
    else:
        train_dataset = CrossDomainDataset(cps_train, bio_train, beh_train)
        val_dataset = CrossDomainDataset(cps_val, bio_val, beh_val)
        test_dataset = CrossDomainDataset(cps_test, bio_test, beh_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def create_baseline_dataloaders(
    data: Dict,
    config: Dict,
    dataset_type: str = 'cps'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_ratio = config['data']['split']['train']
    val_ratio = config['data']['split']['val']
    seed = config['project']['seed']
    batch_size = config['training']['baseline']['batch_size']
    
    windows = data['windows']
    labels = data.get('labels', np.zeros(len(windows)))
    
    n_samples = len(windows)
    indices = np.arange(n_samples)
    
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True
    )
    
    val_size = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=seed,
        shuffle=True
    )
    
    train_windows = torch.FloatTensor(windows[train_idx])
    val_windows = torch.FloatTensor(windows[val_idx])
    test_windows = torch.FloatTensor(windows[test_idx])
    
    train_labels = torch.LongTensor(labels[train_idx])
    val_labels = torch.LongTensor(labels[val_idx])
    test_labels = torch.LongTensor(labels[test_idx])
    
    from torch.utils.data import TensorDataset
    
    train_dataset = TensorDataset(train_windows, train_labels)
    val_dataset = TensorDataset(val_windows, val_labels)
    test_dataset = TensorDataset(test_windows, test_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader