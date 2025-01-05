import numpy as np
import torch
import pickle

### import packages from sklearn
from sklearn.decomposition import PCA
from sklearn.svm import SVC

### import packages from self-defined model
from model.mlp import MLPClassifier
from model.lr import LogisticRegression
from model.svm import MultiClassKernelSVMClassifier

import argparse
import time

def main(args):
    with open('data/train_feature.pkl', 'rb') as f:
        X_train = pickle.load(f).toarray()
        
    if args.preprocess:
        X_train = np.sqrt(1-(1-X_train)**2)
    
    y_train = np.load('data/train_labels.npy')
    
    label_distribution = []
    for i in range(20):
        label_distribution.append((y_train == i).sum().item())
        
    print("Label distribution:", label_distribution)
    
        
    with open('data/test_feature.pkl', 'rb') as f:
        X_test = pickle.load(f).toarray()
        
    if args.preprocess:
        X_test= np.sqrt(1-(1-X_test)**2)
    
    if args.use_pca:
        if args.load_pca:
            X_train = np.load(f"PCA_feat/test/X_train_pca_{args.pca_dim}.npy")
            X_test= np.load(f"PCA_feat/test/X_test_pca_{args.pca_dim}.npy")
            print("PCA reduced feature has been loaded. The feature dimension is", X_train.shape[1])
        else:
            num_dim = args.pca_dim
            
            pca = PCA(n_components=num_dim)
            X_train = pca.fit_transform(X_train)
            cumulative_explained_variance = np.sum(pca.explained_variance_ratio_)
            print("Cumulative explained variance ratio:", cumulative_explained_variance)
            
            X_test = pca.transform(X_test)
            
            if args.save_pca:
                np.save(f"PCA_feat/test/X_train_pca_{args.pca_dim}.npy", X_train)
                np.save(f"PCA_feat/test/X_test_pca_{args.pca_dim}.npy", X_test)

    torch.manual_seed(2333)
    if args.model == 'sklearn-SVM':
        model = SVC(kernel='linear')
        # model = SVC(kernel='rbf')
    elif args.model == 'SVM':
        model = MultiClassKernelSVMClassifier(kernel='linear')
        # model = MultiClassKernelSVMClassifier(kernel='rbf', gamma=20)
    elif args.model == 'LR':
        model = LogisticRegression(penalty='l1', solver='liblinear')
    elif args.model == 'MLP':
        MLP_params = {
            'input_size': X_train.shape[1],
            'output_size': 20,
            'hidden_size': 448,
            'drop_rate': 0.95,
            'weight_decay': 3e-8,
            'lr': 1e-3,
            'batch_size': 256,
            'epoch_num': 75,
        }
        model = MLPClassifier(**MLP_params)
        print(f"MLP model with args {MLP_params}")
    else:
        raise ValueError('model not supported')
        
    if args.saved_model is None:
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Train time: {end_time - start_time:.2f}s")
        
        train_accuracy = model.score(X_train, y_train)
        print(f"Accuracy on training set: {train_accuracy:.4f}")
    else:
        model.load_state_dict(torch.load(args.saved_model))
        print(f"Model has been loaded from {args.saved_model}")

    y_test = model.predict(X_test)
    with open(f'submission/res_{args.model}.csv', 'w') as f:
        f.write('ID,label\n')
        for i in range(len(y_test)):
            f.write(f'{i},{y_test[i]}\n')
    
    label_distribution = []
    for i in range(20):
        label_distribution.append((y_test == i).sum().item())
        
    print("Label distribution:", label_distribution)
            
    print(f"Test result has been saved to submission/res_{args.model}.csv")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='Model to use (default: MLP)')
    parser.add_argument('--use_pca', action='store_true', help='Enable PCA (default: False)')
    parser.add_argument('--pca_dim', type=int, default=500, help='PCA dimension (default: 500)')
    parser.add_argument('--load_pca', action='store_true', help='Load PCA feature (default: False)')
    parser.add_argument('--save_pca', action='store_true', help='Save PCA feature (default: False)')
    parser.add_argument('--saved_model', type=str, default=None, help='Use saved model (default: None)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data (default: False)')
    args = parser.parse_args()
    main(args)