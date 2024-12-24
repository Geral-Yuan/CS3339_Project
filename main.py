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

    # X = np.sqrt(X) # 0.925049, 80
    a = 1.0 # 0.925845, 70
    # a = 1.1 # 0.926286, 60
    # a = 1.3 # 0.926286, 70/75
    # a = 1.5 # 0.925933, 75
    X = np.sqrt(1-(1-X/a)**2)
    
    
    # X = X ** 0.25 # 0.924784, 80
    
    # import warnings
    # warnings.filterwarnings("ignore", category=FutureWarning)
    
    # from imblearn.over_sampling import BorderlineSMOTE
    # borderline_smote = BorderlineSMOTE(random_state=42)
    # print("Original data shape:", X.shape, y.shape)
    # X, y = borderline_smote.fit_resample(X, y)
    # print("BorderlineSMOTE data shape:", X.shape, y.shape)
    
    # from imblearn.over_sampling import SMOTE
    # random_state = 114514
    # print("SMOTE random state:", random_state)
    # print("Original data shape:", X.shape, y.shape)
    # X, y = SMOTE(random_state=random_state).fit_resample(X, y)
    # print("SMOTE data shape:", X.shape, y.shape)

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
    args = parser.parse_args()
    log_path = "logs/"+args.log_file
    log_file = open(log_path, "a")
    sys.stdout = Tee(log_file)
    print("-"*100)
    print(f"{timestamp} run with arguments: {args}")
    main(args)
    sys.stdout = sys.__stdout__
    log_file.close()