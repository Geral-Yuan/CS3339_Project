import numpy as np
import pickle
import os

from validation import train_val

import argparse
import sys
from datetime import datetime

class Tee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout  # 保存原始 stdout

    def write(self, message):
        self.file.write(message)  # 写入文件
        self.stdout.write(message)  # 输出到终端

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def main(args):
    with open('data/train_feature.pkl', 'rb') as f:
        X = pickle.load(f).toarray()
    
    y = np.load('data/train_labels.npy')

    if args.preprocess:
        X = np.sqrt(1-(1-X)**2)

    train_val(args, X, y)

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='Model to use (default: MLP)')
    parser.add_argument('--use_pca', action='store_true', help='Enable PCA (default: False)')
    parser.add_argument('--pca_dim', type=int, default=500, help='PCA dimension (default: 500)')
    parser.add_argument('--load_pca', action='store_true', help='Load PCA feature (default: False)')
    parser.add_argument('--save_pca', action='store_true', help='Save PCA feature (default: False)')
    parser.add_argument('--log_file', type=str, default="log.log", help='Log file (default: log.txt)')
    parser.add_argument('--result_file', type=str, default=f"{timestamp}.csv", help='Result file (default: result.txt)')
    parser.add_argument('--timestamp', type=str, default=timestamp, help='Timestamp (default: current time get from system)')
    parser.add_argument('--save_model', action='store_true', help='Save model with highest accuracy on the validation set (default: False)')
    parser.add_argument('--val_when_train', action='store_true', help='Validate when training (default: False)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data (default: False)')
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    if args.save_model:
        os.makedirs("saved_models", exist_ok=True)
    log_path = "logs/"+args.log_file
    log_file = open(log_path, "a")
    sys.stdout = Tee(log_file)
    print("-"*100)
    print(f"{timestamp} run with arguments: {args}")
    main(args)
    sys.stdout = sys.__stdout__
    log_file.close()