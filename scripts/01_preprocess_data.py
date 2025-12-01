import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
from src.utils.config import Config, set_seed
from src.utils.logging import get_logger
from src.data.preprocessing import (
    SWaTPreprocessor,
    WADIPreprocessor,
    WESADPreprocessor,
    SWELLPreprocessor
)


def main():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = Config(config_path)
    
    set_seed(config.config['project']['seed'])
    
    log_dir = config.paths['experiments'] / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('preprocessing', log_dir)
    
    logger.info("="*60)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("="*60)
    
    processed_dir = config.paths['processed']
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n--- Processing SWaT Dataset ---")
    try:
        swat_processor = SWaTPreprocessor(config.config)
        swat_data = swat_processor.process()
        
        logger.info(f"SWaT Train Windows: {swat_data['train_windows'].shape}")
        logger.info(f"SWaT Test Windows: {swat_data['test_windows'].shape}")
        logger.info(f"SWaT Features: {len(swat_data['feature_names'])}")
        
        np.savez(
            processed_dir / 'swat_processed.npz',
            train_windows=swat_data['train_windows'],
            train_labels=swat_data['train_labels'],
            train_severity=swat_data['train_severity'],
            test_windows=swat_data['test_windows'],
            test_labels=swat_data['test_labels'],
            test_severity=swat_data['test_severity'],
            mean=swat_data['mean'],
            std=swat_data['std'],
            feature_names=swat_data['feature_names']
        )
        
        logger.info("✓ SWaT data saved successfully")
        
    except Exception as e:
        logger.error(f"✗ Error processing SWaT: {e}")
        raise
    
    logger.info("\n--- Processing WADI Dataset ---")
    try:
        wadi_processor = WADIPreprocessor(config.config)
        wadi_data = wadi_processor.process()
        
        logger.info(f"WADI Windows: {wadi_data['windows'].shape}")
        logger.info(f"WADI Features: {len(wadi_data['feature_names'])}")
        
        np.savez(
            processed_dir / 'wadi_processed.npz',
            windows=wadi_data['windows'],
            labels=wadi_data['labels'],
            severity=wadi_data['severity'],
            mean=wadi_data['mean'],
            std=wadi_data['std'],
            feature_names=wadi_data['feature_names']
        )
        
        logger.info("✓ WADI data saved successfully")
        
    except Exception as e:
        logger.error(f"✗ Error processing WADI: {e}")
        raise
    
    logger.info("\n--- Processing WESAD Dataset ---")
    try:
        wesad_processor = WESADPreprocessor(config.config)
        wesad_data = wesad_processor.process()
        
        logger.info(f"WESAD Windows: {wesad_data['windows'].shape}")
        logger.info(f"WESAD Features: {wesad_data['windows'].shape[-1]}")
        
        np.savez(
            processed_dir / 'wesad_processed.npz',
            windows=wesad_data['windows'],
            labels=wesad_data['labels'],
            stress=wesad_data['stress'],
            mean=wesad_data['mean'],
            std=wesad_data['std']
        )
        
        logger.info("✓ WESAD data saved successfully")
        
    except Exception as e:
        logger.error(f"✗ Error processing WESAD: {e}")
        raise
    
    logger.info("\n--- Processing SWELL Dataset ---")
    try:
        swell_processor = SWELLPreprocessor(config.config)
        swell_data = swell_processor.process()
        
        if len(swell_data['windows']) > 0:
            logger.info(f"SWELL Windows: {swell_data['windows'].shape}")
            logger.info(f"SWELL Features: {swell_data['windows'].shape[-1]}")
            
            np.savez(
                processed_dir / 'swell_processed.npz',
                windows=swell_data['windows'],
                labels=swell_data['labels'],
                workload=swell_data['workload'],
                mean=swell_data['mean'],
                std=swell_data['std']
            )
            
            logger.info("✓ SWELL data saved successfully")
        else:
            logger.warning("! SWELL dataset is empty - skipping")
        
    except Exception as e:
        logger.error(f"✗ Error processing SWELL: {e}")
        logger.warning("Continuing without SWELL data...")
    
    logger.info("\n--- Creating Combined CPS Dataset ---")
    cps_train = swat_data['train_windows']
    cps_test = swat_data['test_windows']
    cps_train_labels = swat_data['train_labels']
    cps_test_labels = swat_data['test_labels']
    cps_train_severity = swat_data['train_severity']
    cps_test_severity = swat_data['test_severity']
    
    if len(wadi_data['windows']) > 0:
        cps_all = np.vstack([cps_train, cps_test, wadi_data['windows']])
        cps_labels = np.concatenate([cps_train_labels, cps_test_labels, wadi_data['labels']])
        cps_severity = np.concatenate([cps_train_severity, cps_test_severity, wadi_data['severity']])
    else:
        cps_all = np.vstack([cps_train, cps_test])
        cps_labels = np.concatenate([cps_train_labels, cps_test_labels])
        cps_severity = np.concatenate([cps_train_severity, cps_test_severity])
    
    logger.info(f"Combined CPS Windows: {cps_all.shape}")
    
    np.savez(
        processed_dir / 'processed_cps.npz',
        windows=cps_all,
        labels=cps_labels,
        severity=cps_severity,
        mean=swat_data['mean'],
        std=swat_data['std']
    )
    
    logger.info("✓ Combined CPS data saved")
    
    logger.info("\n--- Creating Combined Bio Dataset ---")
    
    if len(wesad_data['windows']) > 0:
        bio_windows = wesad_data['windows']
        bio_labels = wesad_data['labels']
        bio_stress = wesad_data['stress']
        bio_mean = wesad_data['mean']
        bio_std = wesad_data['std']
        
        logger.info(f"Bio Windows: {bio_windows.shape}")
        
        np.savez(
            processed_dir / 'processed_bio.npz',
            windows=bio_windows,
            labels=bio_labels,
            stress=bio_stress,
            mean=bio_mean,
            std=bio_std
        )
        
        logger.info("✓ Bio data saved")
    else:
        logger.warning("! No bio data available")
    
    logger.info("\n--- Creating Combined Behavior Dataset ---")
    
    if len(swell_data['windows']) > 0:
        beh_windows = swell_data['windows']
        beh_labels = swell_data['labels']
        beh_workload = swell_data['workload']
        beh_mean = swell_data['mean']
        beh_std = swell_data['std']
        
        logger.info(f"Behavior Windows: {beh_windows.shape}")
        
        np.savez(
            processed_dir / 'processed_beh.npz',
            windows=beh_windows,
            labels=beh_labels,
            workload=beh_workload,
            mean=beh_mean,
            std=beh_std
        )
        
        logger.info("✓ Behavior data saved")
    else:
        logger.warning("! No behavior data available - creating dummy data")
        
        dummy_beh_windows = np.random.randn(len(bio_windows), 256, 12).astype(np.float32)
        dummy_beh_labels = np.zeros(len(bio_windows))
        dummy_beh_workload = np.random.randint(0, 3, len(bio_windows))
        
        np.savez(
            processed_dir / 'processed_beh.npz',
            windows=dummy_beh_windows,
            labels=dummy_beh_labels,
            workload=dummy_beh_workload,
            mean=np.zeros(12),
            std=np.ones(12)
        )
        
        logger.info("✓ Dummy behavior data created")
    
    logger.info("\n--- Summary ---")
    logger.info(f"Processed data saved to: {processed_dir}")
    logger.info(f"- processed_cps.npz: {cps_all.shape}")
    if len(wesad_data['windows']) > 0:
        logger.info(f"- processed_bio.npz: {bio_windows.shape}")
    if len(swell_data['windows']) > 0:
        logger.info(f"- processed_beh.npz: {beh_windows.shape}")
    
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()