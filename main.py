import numpy as np
import pickle

### import packages from sklearn
from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier

### import packages from self-defined model
from model.mlp import MLPClassifier

import time
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
        
    with open('data/test_feature.pkl', 'rb') as f:
        X_test = pickle.load(f).toarray()
        
    if args.test:
        X_train = X
        y_train = y
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
    train_set_size = X_train.shape[0]
    
    if args.use_pca:
        if args.load_pca:
            X_train = np.load(f"X_train_pca_{args.pca_dim}_{train_set_size}.npy")
            if args.test:
                X_test= np.load(f"X_test_pca_{args.pca_dim}.npy")
            else:
                X_val = np.load(f"X_val_pca_{args.pca_dim}.npy")
            print("PCA reduced feature has been loaded. The feature dimension is", X_train.shape[1])
        else:
            num_dim = args.pca_dim
            
            pca = PCA(n_components=num_dim)
            X_train = pca.fit_transform(X_train)
            cumulative_explained_variance = np.sum(pca.explained_variance_ratio_)
            print("Cumulative explained variance ratio:", cumulative_explained_variance)
            
            if args.test:
                X_test = pca.transform(X_test)
            else:
                X_val = pca.transform(X_val)
            
            if args.save_pca:
                np.save(f"X_train_pca_{args.pca_dim}_{train_set_size}.npy", X_train)
                if args.test:
                    np.save(f"X_test_pca_{args.pca_dim}.npy", X_test)
                else:
                    np.save(f"X_val_pca_{args.pca_dim}.npy", X_val)
    
    if args.model == 'RndFrst':
        model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    elif args.model == 'SVM-l':
        model = SVC(kernel='linear')
    elif args.model == 'SVM-r':
        model = SVC(kernel='rbf')
    elif args.model == 'SVM-s':
        model = SVC(kernel='sigmoid', coef0=1)
    elif args.model == 'LogReg':
        model = LogisticRegression(penalty='l1', solver='liblinear')
    elif args.model == 'KNN':
        model = KNeighborsClassifier(n_neighbors=15)
    elif args.model == 'MLP':
        MLP_args = {
            'input_size': X_train.shape[1],
            'output_size': 20,
            'hidden_size': 1024,
            'drop_rate': 0.9,
            'weight_decay': 1e-7,
            'lr': 1e-3,
            'epoch_num': 20
        }
        model = MLPClassifier(**MLP_args)
        print(f"MLP model with args {MLP_args}")
        # model = MLPClassifier(hidden_layer_sizes=(100),    # 隐藏层
        #                         activation='relu',         # 激活函数：ReLU
        #                         solver='adam',             # 优化器：Adam
        #                         max_iter=200,              # 最大迭代次数
        #                         random_state=42,
        #                         alpha=0.05)                # 正则化参数
    else:
        raise ValueError('model not supported')
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Train time: {end_time - start_time:.2f}s")
    
    start_time = time.time()
    train_accuracy = model.score(X_train, y_train)
    end_time = time.time()
    print(f"Prediction time on training set: {end_time - start_time:.2f}s")
    print(f"Accuracy on training set: {train_accuracy:.4f}")
    
    if args.test:
        y_test = model.predict(X_test)
        with open(f'submission/res_{args.model}.csv', 'w') as f:
            f.write('ID,label\n')
            for i in range(len(y_test)):
                f.write(f'{i},{y_test[i]}\n')
    else:
        start_time = time.time()
        val_accuracy = model.score(X_val, y_val)
        end_time = time.time()
        print(f"Prediction time on validation set: {end_time - start_time:.2f}s")
        print(f"Accuracy on validation set: {val_accuracy:.4f}")
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='Model to use (default: MLP)')
    parser.add_argument('--use_pca', action='store_true', help='Enable PCA (default: False)')
    parser.add_argument('--pca_dim', type=int, default=500, help='PCA dimension (default: 500)')
    parser.add_argument('--load_pca', action='store_true', help='Load PCA feature (default: False)')
    parser.add_argument('--save_pca', action='store_true', help='Save PCA feature (default: False)')
    parser.add_argument('--test', action='store_true', help='Run on test set and output csv file (default: False)')
    parser.add_argument('--log_file', type=str, default="log.txt", help='Log file (default: log.txt)')
    args = parser.parse_args()
    log_file = open(args.log_file, "a")
    sys.stdout = Tee(log_file)
    print("----------------------------------------------------------------------------------------------------")
    print(f"{datetime.now()} run with arguments: {args}")
    main(args)
    sys.stdout = sys.__stdout__
    log_file.close()