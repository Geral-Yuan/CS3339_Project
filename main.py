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
    
    y = np.load('data/train_labels.npy')
    
    # default 0.91223
    
    # X = X[:, X.max(axis=0) > 1e-6]

    X = np.sqrt(X) # 0.92125 | 0.92842
    # X = X ** 0.25 # 0.92143 | 0.92858
    # X = np.sqrt(1-(1-X)**2) # 0.92213 | 0.92842
    # X[X > 1e-6] = -1/np.log(X[X > 1e-6]) # 0.91957
    
    # offset = 0
    # scale = 1
    # N = 40
    # for i in range(N):
    #     X[(X > i/N)&(X < (i+1)/N)] = (X[(X > i/N)&(X < (i+1)/N)] - i/N) * scale + offset
    #     offset += scale / N
    #     scale -= 1 / N
        
    # X = np.log(X+3e-3)
    # X -= X.min(axis=0)
    # X /= X.max(axis=0)
    
    # X = np.log1p(X) # 0.91302
    # X = np.log1p(X*np.e) # 0.91904
    
    # import warnings
    # warnings.filterwarnings("ignore", category=FutureWarning)
    
    # from imblearn.over_sampling import BorderlineSMOTE
    # borderline_smote = BorderlineSMOTE(random_state=42)
    # print("Original data shape:", X.shape, y.shape)
    # X, y = borderline_smote.fit_resample(X, y)
    # print("BorderlineSMOTE data shape:", X.shape, y.shape)
    
    # from imblearn.over_sampling import SMOTE
    # print("Original data shape:", X.shape, y.shape)
    # X, y = SMOTE(random_state=42).fit_resample(X, y)
    # print("SMOTE data shape:", X.shape, y.shape)
    
    # for i in range(3):
    #     if i == 0:
    #         X = np.sqrt(X)
    #     elif i == 1:
    #         X = X ** 0.25
    #     elif i == 2:
    #         X = np.sqrt(1-(1-X)**2)
    #     train_val(args, X, y, i)
    #     with open('data/train_feature.pkl', 'rb') as f:
    #         X = pickle.load(f).toarray()

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
    print("-"*100)
    print(f"{timestamp} run with arguments: {args}")
    main(args)
    sys.stdout = sys.__stdout__
    log_file.close()