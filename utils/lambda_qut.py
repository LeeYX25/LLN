import torch
import numpy as np

def function_derivative(func, u):
    y = func(u)
    y.backward()
    return u.grad.item()

def lambda_qut_classification(X, hat_p, act_fun,tau,weight,n_samples=10000, mini_batch_size=500, alpha=0.05, option='quantile'):
    offset = 0 if n_samples % mini_batch_size == 0 else 1
    n_samples_per_batch = n_samples // mini_batch_size + offset
    
    n, p1 = X.shape    
    fullList = torch.zeros((mini_batch_size*n_samples_per_batch,), device= X.device)
    num_classes =len(hat_p)
    m=num_classes

    for index in range(n_samples):
        y_sample = torch.multinomial(hat_p, num_samples=n , replacement=True)
        y_sample = torch.nn.functional.one_hot(y_sample, num_classes=num_classes).float().to('cuda')
        W=transweight(weight,tau,n).to('cuda')
        ybar= get_ybar_cla(y_sample,tau,weight)
        ybar=ybar.repeat(n,1).to('cuda')
        Y_diff=(y_sample-ybar).to('cuda')
        temp = torch.matmul(W, Y_diff)
        n, p = X.shape
        result = torch.matmul(X.T, temp) 
        col_sums = torch.sum(torch.abs(result), dim=1)  
        max_val, max_idx = torch.max(col_sums, dim=0)
        fullList[index] = max_val
    fullList = fullList * function_derivative(act_fun, torch.tensor(0, dtype=torch.float, requires_grad=True, device = X.device))
 
    if option=='full':
        return fullList
    elif option=='quantile':
        return torch.quantile(fullList, 1-alpha)
    else:
        pass


def lambda_qut_regression(X,act_fun,tau, weight, n_samples=10000, mini_batch_size=500, alpha=0.05, option='quantile'):
    offset = 0 if n_samples % mini_batch_size == 0 else 1
    n_samples_per_batch = n_samples // mini_batch_size + offset
    n, p1 = X.shape
    fullList = torch.zeros((mini_batch_size * n_samples_per_batch,), device=X.device)

    for index in range(n_samples):
        y_sample = torch.normal(mean=0., std=1, size=(n, 1)).type(torch.float).to(X.device)
        ybar1 = get_ybar_reg1(y_sample, tau, weight)
        ybar2 = get_ybar_reg2(y_sample, tau, weight)
        xybar = torch.tensor([]).to(X.device)
        for j in range(0, p1):
            x_j = X[:, j]
            x_j = x_j.unsqueeze(1)
            j_matrix = x_j.repeat(1, n)
            j_matrix=j_matrix.t()
            new = torch.sum(j_matrix * ybar1)
            new = new.unsqueeze(0)
            xybar = torch.cat((xybar, new), dim=0)
        xybar_max = torch.amax(torch.abs(xybar), dim=0)
        norms = torch.norm(ybar2, p=2)
        fullList[index] = xybar_max / norms
    fullList = fullList * function_derivative(act_fun,torch.tensor(0, dtype=torch.float, requires_grad=True, device=X.device))
    print(fullList)

    if option == 'full':
        return fullList
    elif option == 'quantile':
        return torch.quantile(fullList, 1 - alpha)
    else:
        pass


def get_ybar_cla(y, tau, weight):
    n = y.shape[0]
    m=y.shape[1]
    chat=torch.zeros(1,m)
    weight = transweight(weight, tau,n).to(y.device)
    for i in range(0,m):
        ymatrix = y[:,i].unsqueeze(1).repeat(1, n)
        c1 = torch.sum(weight * ymatrix)
        c2 = torch.sum(weight)
        chat[0][i]=c1/c2
    return (chat[0])


def get_ybar_reg1(y, tau, weight):
    n = y.shape[0]
    weight = transweight(weight, tau,n).to(y.device)
    c_hat = get_c_hat_reg(y,tau,weight)
    identity_matrix = torch.ones(n, n).to(y.device)
    ymatrix = y.repeat(1, n).to(y.device)
    ybar = (ymatrix - identity_matrix * c_hat)*weight
    return (ybar)


def get_ybar_reg2(y, tau, weight):
    n = y.shape[0]
    weight = transweight(weight, tau,n).to(y.device)
    c_hat = get_c_hat_reg(y,tau,weight)
    identity_matrix = torch.ones(n, n).to(y.device)
    ymatrix = y.repeat(1, n).to(y.device)
    ybar = (ymatrix - identity_matrix * c_hat)**torch.sqrt(weight)
    return (ybar)


def get_c_hat_reg(y, tau, weight):
    n = y.shape[0]
    ymatrix = y.repeat(1, n)
    c1 = torch.sum(weight * ymatrix)
    c2 = torch.sum(weight)
    return (c1/c2)


def transweight(w, tau, t):
    result = torch.empty_like(w)
    diag_mask = torch.eye(w.shape[0], dtype=torch.bool, device=w.device)

    result[diag_mask] = (tau/t) * w[diag_mask]
    result[~diag_mask] = (1 - tau)/(t*(t-1)) * w[~diag_mask]

    return result


