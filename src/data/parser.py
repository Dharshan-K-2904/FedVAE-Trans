import os
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

def parse_hdfs_logs(raw_log_path: str, output_dir: str):
    config = TemplateMinerConfig()
    config.load(os.path.dirname(__file__) + "/drain3.ini")
    template_miner = TemplateMiner(config=config)
    
    parsed_sequences = []
    print("Initiating Drain3 Log Parsing...")
    with open(raw_log_path, 'r') as f:
        for line in f:
            log_message = line.strip().split(' ', 5)[-1] 
            result = template_miner.add_log_message(log_message)
            parsed_sequences.append(result["cluster_id"])
            
    window_size = 128
    windows = [parsed_sequences[i:i+window_size] 
               for i in range(0, len(parsed_sequences), window_size) 
               if len(parsed_sequences[i:i+window_size]) == window_size]
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'sequences.npy'), np.array(windows))
    print(f"Parsing complete. Extracted {len(template_miner.drain.clusters)} templates.")
