import numpy as np
import pandas as pd
import torch
import os

### import packages from sklearn
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from itertools import product

### import packages from self-defined model
from model.mlp import MLPClassifier
from model.lr import LogisticRegression
from model.svm import MultiClassKernelSVMClassifier

import time

def cross_val(args, X, y, params, k_folds=5):
    print(f"Cross validation with parameters: {params}")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    train_accuracy_list = []
    val_accuracy_list = []
    best_accuracy = 0
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

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
                    os.makedirs('PCA_feat/val', exist_ok=True)
                    np.save(f"PCA_feat/val/X_train_pca_{k_folds}folds_{args.pca_dim}_{fold}.npy", X_train)
                    np.save(f"PCA_feat/val/X_val_pca_{k_folds}folds_{args.pca_dim}_{fold}.npy", X_val)

        # 404, 2333, 3407
        torch.manual_seed(2333)
        if args.model == 'sklearn-SVM':
            model = SVC(**params)
        elif args.model == 'SVM':
            model = MultiClassKernelSVMClassifier(**params)
        elif args.model == 'LR':
            model = LogisticRegression(num_classes=20, **params)
        elif args.model == 'MLP':
            model = MLPClassifier(input_size=X_train.shape[1], output_size=20, **params)
        else:
            raise ValueError('model not supported')
        
        if args.val_when_train:
            if args.model != 'MLP':
                raise ValueError('Validation during training is only supported for MLP')
            print(f"\n\nValidation during training fold {fold+1}/{k_folds}:")
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        train_accuracy_list.append(train_accuracy)
        val_accuracy = model.score(X_val, y_val)
        val_accuracy_list.append(val_accuracy)
        if args.save_model:
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), f"saved_models/{args.model}_{args.timestamp}.pt")
        print(f"Fold {fold+1}/{k_folds}, Train accuracy: {train_accuracy:.6f}, Validation accuracy: {val_accuracy:.6f}")
    
    return np.mean(train_accuracy_list), np.mean(val_accuracy_list)
        
        

def grid_search(args, X, y, params):
    keys = list(params.keys())
    values = [params[key] if isinstance(params[key], list) else [params[key]] for key in keys]
    
    best_params = None
    best_score = 0
    results_df= pd.DataFrame(columns=list(params.keys())+['train_accuracy', 'val_accuracy'])
    for combination in product(*values):
        param_combination = dict(zip(keys, combination))
        train_acc, val_acc = cross_val(args, X, y, param_combination)
        print(f"Train accuracy: {train_acc:.6f}, Validation accuracy: {val_acc:.6f}")
        if val_acc > best_score:
            best_score = val_acc
            best_params = param_combination
        new_row = pd.Series({**param_combination, 'train_accuracy': train_acc, 'val_accuracy': val_acc})
        if results_df.empty:
            results_df = new_row.to_frame().T
        else:
            results_df = pd.concat([results_df, new_row.to_frame().T], ignore_index=True)

    results_df.to_csv(f"results/{args.result_file}", index=False)
    print(f"Best validation accuracy: {best_score}, Best params: {best_params}")
        

        

def train_val(args, X, y):
    if args.model == 'sklearn-SVM':
        params = {
            'kernel': ['linear', 'rbf'],
        }
    elif args.model == 'SVM':
        params = {
            'kernel': 'linear'
        }
        # params = {
        #     'kernel': 'rbf',
        #     'gamma': 20,
        # }
    elif args.model == 'LR':
        # params = {
        #     'lr': [50, 75, 100, 125, 150],
        #     'num_iterations': [500, 750, 1000]
        # }
        params = {
            'lr': 50,
            'num_iterations': 750
        }
    elif args.model == 'MLP':
        # params = {
        #     'hidden_size': [448, 512, 576, 640],
        #     'drop_rate': [0.6, 0.8, 0.9, 0.95],
        #     'weight_decay': [3e-7, 1e-7, 3e-8, 1e-8, 0],
        #     'lr': 1e-3,
        #     'batch_size': 256,
        #     'epoch_num': [45, 60, 75],
        #     'resNet': [True, False]
        # }
        params = {
            'hidden_size': 448,
            'drop_rate': 0.95,
            'weight_decay': 3e-8,
            'lr': 1e-3,
            'batch_size': 256,
            'epoch_num': 75,
        }
        
    else:
        raise ValueError('model not supported')
    
    print(f"Grid search with parameters: {params}")
    start_time = time.time()
    grid_search(args, X, y, params)
    end_time = time.time()
    print(f"Total grid search time: {end_time - start_time:.2f}s")