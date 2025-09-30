import numpy as np
import torch
import torch.nn as nn
import utils


def lambda_qut_regression_linear(X, n_samples=10000, mini_batch_size=500, alpha=0.05, option='quantile'):
    offset = 0 if n_samples % mini_batch_size == 0 else 1
    n_samples_per_batch = n_samples // mini_batch_size + offset
    
    n, p1 = X.shape    
    fullList = torch.zeros((mini_batch_size*n_samples_per_batch,), device= X.device)

    for index in range(n_samples_per_batch):
        y_sample = torch.normal(mean=0., std=1, size=(n, 1, mini_batch_size)).type(torch.float).to(X.device)
        y = (torch.mean(y_sample, dim=0)- y_sample)
        xy = y * X.unsqueeze(2).expand(-1, -1, mini_batch_size)
        xy_max = torch.amax(torch.abs(torch.sum(xy, dim=0)), dim=0)
        norms = torch.norm(y, p=2, dim=0)
        fullList[index * mini_batch_size:(index+1) * mini_batch_size] = (xy_max / norms).detach()

    if option=='full':
        return fullList
    elif option=='quantile':
        return torch.quantile(fullList, 1-alpha)
    else:
        pass


class LinearRegressionLoss(nn.Module):
    def __init__(self, lambda_, nu):
        super().__init__()
        self.lambda_ = lambda_
        self.nu = nu
        self.mse_loss = nn.MSELoss(reduction='sum')

    def custom_penalty_fun(self, theta):
        abs_theta = torch.abs(theta.weight)
        power_theta = abs_theta.pow(1 - self.nu)
        penalty_weight = torch.sum(abs_theta.div(1 + power_theta))
        
        abs_theta = torch.abs(theta.bias)
        power_theta = abs_theta.pow(1 - self.nu)
        penalty_bias = torch.sum(abs_theta.div(1 + power_theta))

        return penalty_weight + penalty_bias

    def forward(self, input, target, theta):
        mse_loss_value = self.mse_loss(input, target)
        square_root_lasso_loss = torch.sqrt(mse_loss_value)

        if self.nu < 1:
            regularization = self.lambda_ * self.custom_penalty_fun(theta)
        else:
            regularization = self.lambda_ * (torch.sum(torch.abs(theta.weight)) + torch.sum(torch.abs(theta.bias)))

        total_loss = square_root_lasso_loss + regularization
        return total_loss, square_root_lasso_loss


class LinearModel():
    def __init__(self, penalty=0, lambda_qut=None, device=None):
        self.penalty = penalty
        self.lambda_qut = lambda_qut
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') if device is None else device

    def fit(self, X, y, verbose=False):
        # Data Processing
        X, y = utils.data_to_tensor(X, y)
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Model Parameters
        self.theta = nn.Linear(X.shape[1], 1, dtype=torch.float, device=self.device)

        # Lambda qut
        self.lambda_qut = lambda_qut_regression_linear(X) if self.lambda_qut is None else self.lambda_qut
        
        # Training
        for i in range(-1, 6):
            nu = 1 if self.penalty == 1 else [1, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1][i + 1]
            lambi = self.lambda_qut * (np.exp(i) / (1 + np.exp(i)) if i < 5 else 1)
            rel_err = 1e-9 if i == 5 else 1e-5
            init_lr = 0.1*0.8**(i+1)

            if verbose:
                print(f"Lambda = {lambi.item():.4f} -- Nu = {nu}" if self.penalty == 0 else f"Lambda = {lambi.item():.4f}")
            
            self.train_model(nu, X, y, lambi, init_lr, rel_err, verbose)

        # Check if returning mean(y) gives better results
        model_predictions = self.forward(X)
        if self.penalty == 0:
            loss_fn = LinearRegressionLoss(self.lambda_qut, 0.1).to(self.device)
        else:
            loss_fn = LinearRegressionLoss(self.lambda_qut, 1).to(self.device)
        model_error = loss_fn(model_predictions, y, self.theta)[0]

        metric = torch.nn.MSELoss(reduction='sum')
        baseline_error = torch.sqrt(metric(y.mean(), y))

        if model_error > baseline_error:
            self.theta.weight.data.fill_(0)
            self.theta.bias.data.fill_(y.mean().item())

        
        self.important_features = self.imp_feat()
        if verbose: print("MODEL FITTED !")

    def forward(self, X):
        return self.theta(X).squeeze()

    def predict(self, X):
        X = utils.X_to_tensor(X).to(self.device)
        return self.forward(X).detach().cpu().numpy()

    def imp_feat(self):
        features = torch.nonzero(self.theta.weight.squeeze(), as_tuple=False).squeeze()
        if features.ndim == 0: features = [features.item()]
        else: features = features.tolist()
        return len(features), features


    def train_model(self, nu, X, y, lambda_, init_lr, rel_err, verbose):
        loss_fn = LinearRegressionLoss(lambda_, nu).to(self.device)
        
        epoch, last_loss = 0, np.inf
        optimizer = utils.FISTA(params=self.theta.parameters(), lr=init_lr, lambda_=lambda_, nu=nu)

        lr_factor = 0.9
        max_epochs = 10000

        while epoch < max_epochs:
            optimizer.zero_grad()

            y_pred = self.forward(X)
            loss, bare_loss = loss_fn(y_pred, y, self.theta)
            loss = loss.detach()

            if verbose and epoch % 20 ==0:
                print(f"\tEpoch: {epoch} | Loss: {loss.item():.5f} | Non zeros parameters: {self.imp_feat()} | learning rate : {optimizer.get_lr():.6f}")

            if loss > last_loss: 
                optimizer.update(optimizer.get_lr()*lr_factor)

            if epoch % 10 == 0:
                if epoch > 0 and abs(loss - last_loss) / loss < rel_err:
                    if verbose: print(f"\n\t Descent stopped: loss is no longer decreasing significantly.\n")
                    break
                last_loss = loss
                
            epoch += 1
            bare_loss.backward()
            optimizer.step()

        if epoch == max_epochs and verbose: print("FISTA descent stopped: maximum iterations reached") 
