import numpy as np
import pickle

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
    
    
    # X = np.sqrt(X) # 0.92134
    # X = X ** 0.25 # 0.91975
    X = np.sqrt(1-(1-X)**2) # 0.92178
    # X = np.log1p(X*np.e) # 0.91877
    # X = np.tanh(X)
    
    y = np.load('data/train_labels.npy')

    train_val(args, X, y)
    

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='Model to use (default: MLP)')
    parser.add_argument('--use_pca', action='store_true', help='Enable PCA (default: False)')
    parser.add_argument('--pca_dim', type=int, default=500, help='PCA dimension (default: 500)')
    parser.add_argument('--load_pca', action='store_true', help='Load PCA feature (default: False)')
    parser.add_argument('--save_pca', action='store_true', help='Save PCA feature (default: False)')
    parser.add_argument('--log_file', type=str, default="log.txt", help='Log file (default: log.txt)')
    parser.add_argument('--result_file', type=str, default=f"{timestamp}.csv", help='Result file (default: result.txt)')
    args = parser.parse_args()
    log_file = open(args.log_file, "a")
    sys.stdout = Tee(log_file)
    print("-"*50)
    print(f"{timestamp} run with arguments: {args}")
    main(args)
    sys.stdout = sys.__stdout__
    log_file.close()