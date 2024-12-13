import numpy as np
import pandas as pd
import torch

### import packages from sklearn
from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold
from itertools import product

### import packages from self-defined model
from model.mlp import MLPClassifier

import time

def cross_val(args, X, y, param, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    train_accuracy_list = []
    val_accuracy_list = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # import warnings
        # warnings.filterwarnings("ignore", category=FutureWarning)
        
        # from imblearn.over_sampling import BorderlineSMOTE
        # borderline_smote = BorderlineSMOTE(random_state=42)
        # print("Original data shape:", X_train.shape, y_train.shape)
        # X_train, y_train = borderline_smote.fit_resample(X_train, y_train)
        # print("BorderlineSMOTE data shape:", X_train.shape, y_train.shape)
        
        # from imblearn.over_sampling import SMOTE
        # print("Original data shape:", X_train.shape, y_train.shape)
        # X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        # print("SMOTE data shape:", X_train.shape, y_train.shape)
        
        if args.use_pca:
            if args.load_pca:
                X_train = np.load(f"PCA_feat/val/X_train_pca_{k_folds}folds_{args.pca_dim}_{fold}.npy")
                X_val= np.load(f"PCA_feat/val/X_val_pca_{k_folds}folds_{args.pca_dim}_{fold}.npy")
                print("PCA reduced feature has been loaded. The feature dimension is", X_train.shape[1])
            else:
                num_dim = args.pca_dim
                
                pca = PCA(n_components=num_dim)
                X_train = pca.fit_transform(X_train)
                cumulative_explained_variance = np.sum(pca.explained_variance_ratio_)
                print("Cumulative explained variance ratio:", cumulative_explained_variance)
                
                X_val = pca.transform(X_val)
                
                if args.save_pca:
                    np.save(f"PCA_feat/val/X_train_pca_{k_folds}folds_{args.pca_dim}_{fold}.npy", X_train)
                    np.save(f"PCA_feat/val/X_val_pca_{k_folds}folds_{args.pca_dim}_{fold}.npy", X_val)

        torch.manual_seed(2333)
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
            model = MLPClassifier(input_size=X_train.shape[1], output_size=20, **param)
        else:
            raise ValueError('model not supported')
        
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        train_accuracy_list.append(train_accuracy)
        val_accuracy = model.score(X_val, y_val)
        val_accuracy_list.append(val_accuracy)
    
    return np.mean(train_accuracy_list), np.mean(val_accuracy_list)
        
        

def grid_search(args, X, y, params, id):
    keys = list(params.keys())
    values = [params[key] if isinstance(params[key], list) else [params[key]] for key in keys]
    
    best_params = None
    best_score = 0
    results_df= pd.DataFrame(columns=list(params.keys())+['train_accuracy', 'val_accuracy', 'func_id'])
    for combination in product(*values):
        param_combination = dict(zip(keys, combination))
        train_acc, val_acc = cross_val(args, X, y, param_combination)
        if val_acc > best_score:
            best_score = val_acc
            best_params = param_combination
        new_row = pd.Series({**param_combination, 'train_accuracy': train_acc, 'val_accuracy': val_acc, 'func_id': id})
        if results_df.empty:
            results_df = new_row.to_frame().T
        else:
            results_df = pd.concat([results_df, new_row.to_frame().T], ignore_index=True)

    results_df.to_csv(f"results/{id}_{args.result_file}", index=False)
    print(f"Best validation accuracy: {best_score}, Best params: {best_params}")
        

        

def train_val(args, X, y, id=0):
    if args.model == 'RndFrst':
        pass
        # model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    elif args.model == 'SVM-l':
        pass
        # model = SVC(kernel='linear')
    elif args.model == 'SVM-r':
        pass
        # model = SVC(kernel='rbf')
    elif args.model == 'SVM-s':
        pass
        # model = SVC(kernel='sigmoid', coef0=1)
    elif args.model == 'LogReg':
        pass
        # model = LogisticRegression(penalty='l1', solver='liblinear')
    elif args.model == 'KNN':
        pass
        # model = KNeighborsClassifier(n_neighbors=15)
    elif args.model == 'MLP':
        # params = {
        #     'hidden_size': [384, 512, 640],
        #     'drop_rate': [0.9, 0.95, 0.98],
        #     'weight_decay': [3e-7, 1e-7, 3e-8],
        #     'lr': [1e-3],
        #     'batch_size': [256],
        #     'epoch_num': [40, 50],
        #     'mask_prob': [0]
        # }
        # params = {
        #     'hidden_size': [512, 1024],
        #     'drop_rate': 0.9,
        #     'weight_decay': [3e-7, 1e-7],
        #     'lr': [1e-3, 5e-4],
        #     'batch_size': [128, 256],
        #     'epoch_num': [30, 50, 100, 200],
        #     'mask_prob': 0
        # }
        params = {
            'hidden_size': 640,
            'drop_rate': 0.95,
            'weight_decay': 3e-8,
            'lr': 1e-3,
            'batch_size': 256,
            'epoch_num': 50,
            'mask_prob': 0
        }
        
        print(f"Grid search with parameters: {params}")
    else:
        raise ValueError('model not supported')
        
    start_time = time.time()
    grid_search(args, X, y, params, id)
    end_time = time.time()
    print(f"Total grid search time: {end_time - start_time:.2f}s")