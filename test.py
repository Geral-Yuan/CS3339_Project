import numpy as np
import torch
import pickle

### import packages from sklearn
from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier

### import packages from self-defined model
from model.mlp import MLPClassifier

import argparse
import time

def main(args):
    with open('data/train_feature.pkl', 'rb') as f:
        X_train = pickle.load(f).toarray()
        
    X_train = np.sqrt(1-(1-X_train)**2)
    
    y_train = np.load('data/train_labels.npy')
    
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    from imblearn.over_sampling import SMOTE

    print("Original data shape:", X_train.shape, y_train.shape)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train) # To figure out good seed
    print("SMOTE data shape:", X_train.shape, y_train.shape)
    
    label_distribution = []
    for i in range(20):
        label_distribution.append((y_train == i).sum().item())
        
    print("Label distribution:", label_distribution)
        
    with open('data/test_feature.pkl', 'rb') as f:
        X_test = pickle.load(f).toarray()
        
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
            'hidden_size': 640,
            'drop_rate': 0.95,
            'weight_decay': 3e-8,
            'lr': 1e-3,
            'batch_size': 256,
            'epoch_num': 50,
            'mask_prob': 0
        }
        model = MLPClassifier(**MLP_args)
        print(f"MLP model with args {MLP_args}")
    else:
        raise ValueError('model not supported')
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Train time: {end_time - start_time:.2f}s")
    
    train_accuracy = model.score(X_train, y_train)
    print(f"Accuracy on training set: {train_accuracy:.4f}")
    
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
    args = parser.parse_args()
    main(args)