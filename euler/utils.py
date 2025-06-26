import sys
from pathlib import Path
import os
import logging 

def add_src_and_logger(is_renkolab, save_dir):
    src_path = Path.home() / "work" / "fco2diffusion" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    if is_renkolab:
        DATA_PATH = '/home/jovyan/work/datapolybox/'
    else:
        DATA_PATH = '../data/training_data/'
    
    if save_dir is None:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return DATA_PATH, None
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    logging.basicConfig(
        filename=save_dir+'training.log',
        filemode='a',
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
        )
    return DATA_PATH, logging
