import torch
from sklearn.base import ClassifierMixin, BaseEstimator
from tqdm import tqdm

class LogisticRegression(ClassifierMixin,BaseEstimator):
    def __init__(self, num_classes=20, lr=1e-3, num_iterations=1000):
        super(LogisticRegression, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.num_iterations = num_iterations
        
    @torch.no_grad()
    def fit(self, X, y):
        num_samples = X.shape[0]
        X = torch.tensor(X, dtype=torch.float32).cuda()
        y = torch.eye(self.num_classes)[y].cuda()
        self.beta = torch.zeros((X.shape[1], self.num_classes), requires_grad=False, device='cuda')
        self.bias = torch.zeros((1, self.num_classes), requires_grad=False, device='cuda')
        for i in tqdm(range(self.num_iterations)):
            # forward pass
            y_linear = torch.mm(X, self.beta) + self.bias
            y_hat = torch.softmax(y_linear, dim=1)
            # backpropagation
            d_beta = torch.mm(X.T, (y_hat - y)) / num_samples
            d_bias = torch.mean(y_hat - y, dim=0)
            # update weights
            self.beta -= self.lr * d_beta
            self.bias -= self.lr * d_bias
            
    @torch.no_grad()
    def predict(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32).cuda()
        y_linear = torch.mm(X_test, self.beta) + self.bias
        y_hat = torch.softmax(y_linear, dim=1)
        return torch.argmax(y_hat, dim=1).cpu().numpy()
    
    @torch.no_grad()
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return (y_pred == y_test).mean().item()
        