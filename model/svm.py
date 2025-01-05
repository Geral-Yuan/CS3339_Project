from sklearn.base import ClassifierMixin,BaseEstimator
from cvxopt import matrix, solvers
import numpy as np

class KernelSVMClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self, kernel='linear', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.w = None
        self.b = None
        self.gamma = gamma
        self.gamma_value = None
        self.alpha = None
        self.SV_X = None
        self.SV_y = None
        
    def kernel_func(self, X1, X2):
        """
        kernel function
        X1: (D, N1)
        X2: (D, N2)
        return: (N1, N2)
        """
        if self.kernel == 'linear':
            return X1.T @ X2
        elif self.kernel == 'rbf':
            X1_squared = np.sum(X1**2, axis=0).reshape(-1, 1)  # (N1, 1)
            X2_squared = np.sum(X2**2, axis=0).reshape(1, -1)  # (1, N2)
            pairwise_dists = X1_squared + X2_squared - 2 * X1.T @ X2  # (N1, N2)
            return np.exp(-self.gamma_value * pairwise_dists)
            # return np.exp(-self.gamma_value * ((X1[:,:,None] - X2[:,None]) ** 2).sum(axis=0))
        else:
            raise NotImplementedError("Kernel not implemented")
    
    def fit(self, X, y):
        N, D = X.shape
        if self.gamma == 'scale':
            self.gamma_value = 1 / (X.var() * D)
        elif self.gamma == 'auto':
            self.gamma_value = 1 / D
        elif isinstance(self.gamma, float) or isinstance(self.gamma, int):
            self.gamma_value = self.gamma
        else:
            raise NotImplementedError("gamma not implemented")
        
        # minimize 1/2 x.T P x + q.T x
        # P = y * y.T * K (K = kernel(X))
        P = matrix(y * y.T * self.kernel_func(X.T, X.T))
        # q = (-1, -1, ..., -1).T
        q = matrix(-1 * np.ones(N))
        
        # object to Gx <= h (0 <= alpha <= C)
        # G = (-I, I).T
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        # h = (0, 0, ..., 0, C, C, ..., C).T
        h = matrix(np.hstack((np.zeros(N), self.C * np.ones(N))))
        
        # object to Ax = b
        A = matrix(y.astype(float), (1, N))
        b = matrix(0.0)
        
        sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
        alpha = np.array(sol['x']).ravel()
        EPSILON = 1e-6
        SV_idx = alpha > EPSILON
        free_SV_idx = (alpha > EPSILON) & (alpha < self.C - EPSILON)
        
        if self.kernel == 'linear':
            self.w = X.T @ (alpha * y)
            self.b = np.median(y[free_SV_idx] - self.kernel_func(X[free_SV_idx,:].T,self.w[:,None]).squeeze())
        else:
            self.SV_alpha = alpha[SV_idx]
            self.SV_X = X[SV_idx,:]
            self.SV_y = y[SV_idx]
            b_list = []
            for idx in np.argwhere(free_SV_idx):
                b_list.append(y[idx] - (self.SV_alpha*self.SV_y*self.kernel_func(self.SV_X.T, X[idx,:].T)).sum())
            self.b = np.median(np.array(b_list))
            
        
    def predict(self, X):
        if self.kernel == 'linear':
            return np.sign(self.kernel_func(X.T, self.w[:,None]).squeeze() + self.b)
        else:
            return np.sign((self.kernel_func(X.T, self.SV_X.T)*self.SV_alpha.reshape(1,-1)*self.SV_y.reshape(1,-1) + self.b).sum(axis=1))
    
    def score(self, X, y):
        return (self.predict(X) == y).mean()
    
    def decision_function(self, X):
        if self.kernel == 'linear':
            return self.kernel_func(X.T, self.w[:,None]).squeeze() + self.b
        else:
            return (self.kernel_func(X.T, self.SV_X.T)*self.SV_alpha.reshape(1,-1)*self.SV_y.reshape(1,-1) + self.b).sum(axis=1)
  
    
class MultiClassKernelSVMClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self, kernel='linear', C=1.0, gamma='scale', decision_function_shape='ovr'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.decision_function_shape = decision_function_shape
        self.SVMs = None
        
    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        if self.decision_function_shape == 'ovr':
            self.SVMs = [KernelSVMClassifier(kernel=self.kernel, C=self.C, gamma=self.gamma) for _ in range(self.num_classes)]
            for i in range(self.num_classes):
                y_binary = np.where(y == i, 1, -1)
                self.SVMs[i].fit(X, y_binary)
        else:
            raise NotImplementedError("decision_function_shape not implemented")
        
    def predict(self, X):
        if self.decision_function_shape == 'ovr':
            decision_values = np.array([svm.decision_function(X) for svm in self.SVMs])
            return decision_values.argmax(axis=0)
        else:
            raise NotImplementedError("decision_function_shape not implemented")
        
    def score(self, X, y):
        return (self.predict(X) == y).mean()
