import torch


def get_device(device_str: str = 'auto') -> torch.device:
    
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using device: MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("Using device: CPU")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device_str}")
    
    return device


def get_device_info(device: torch.device) -> dict:
    
    info = {'device_type': str(device)}
    
    if device.type == 'cuda':
        info['device_name'] = torch.cuda.get_device_name(device)
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        info['memory_allocated'] = torch.cuda.memory_allocated(device) / 1024**3
        info['memory_reserved'] = torch.cuda.memory_reserved(device) / 1024**3
    
    elif device.type == 'mps':
        info['device_name'] = 'Apple Silicon'
        info['device_count'] = 1
    
    else:
        info['device_name'] = 'CPU'
        info['device_count'] = 1
    
    return info


def clear_cache(device: torch.device):
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
    elif device.type == 'mps':
        torch.mps.empty_cache()
        print("MPS cache cleared")