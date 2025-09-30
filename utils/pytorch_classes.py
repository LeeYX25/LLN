import torch
import torch.nn as nn
from scipy.optimize import root_scalar, newton
import numpy as np

class Custom_act_fun(nn.Module): 
    def __init__(self, M=20, k=1, u_0=1):
        super(Custom_act_fun, self).__init__() 
        self.M = M
        self.k = k
        self.u_0 = u_0
        self.softplus = nn.Softplus(beta=self.M)
  
    def forward(self, u: torch.Tensor): 
        return (1/self.k) * (self.softplus(u + self.u_0).pow(self.k) - self.softplus(self.u_0 * torch.ones_like(u)).pow(self.k))




class CustomRegressionLoss(nn.Module):
    def __init__(self, lambda_, nu):
        super().__init__()
        self.lambda_ = lambda_
        self.nu = nu
        self.mse_loss = nn.MSELoss(reduction='sum')  



    def custom_penalty_fun(self, layer1):
        abs_weight = torch.abs(layer1.weight).sum()
        power_weight = abs_weight.pow(1-self.nu)
        penalty_weight = abs_weight.div(1+power_weight)

        abs_bias = torch.abs(layer1.bias).sum()
        power_bias = abs_bias.pow(1-self.nu)
        penalty_bias = abs_bias.div(1+power_bias)

        return penalty_weight + penalty_bias

    def forward(self, input, target, layer1):
        mse_loss_value = self.mse_loss(input, target)
        square_root_lasso_loss = torch.sqrt(mse_loss_value)

        if self.nu < 1:
            regularization = self.lambda_ * self.custom_penalty_fun(layer1)
        else:
            regularization = self.lambda_ * (torch.abs(layer1.weight).sum() + torch.abs(layer1.bias).sum())

        total_loss = square_root_lasso_loss + regularization
        return total_loss, square_root_lasso_loss


class CustomClassificationLoss(nn.Module):
    def __init__(self, lambda_, nu):
        super().__init__()
        self.lambda_ = lambda_
        self.nu = nu
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')  

    def custom_penalty_fun(self, layer1):
        abs_weight = torch.abs(layer1.weight).sum()
        power_weight = abs_weight.pow(1-self.nu)
        penalty_weight = abs_weight.div(1+power_weight)

        abs_bias = torch.abs(layer1.bias).sum()
        power_bias = abs_bias.pow(1-self.nu)
        penalty_bias = abs_bias.div(1+power_bias)

        return penalty_weight + penalty_bias

    def forward(self, input, target, layer1):
        cross_entropy_loss_value = self.cross_entropy_loss(input, target)

        if self.nu < 1:
            regularization = self.lambda_ * self.custom_penalty_fun(layer1)
        else:
            regularization = self.lambda_ * (torch.abs(layer1.weight).sum() + torch.abs(layer1.bias).sum())

        total_loss = cross_entropy_loss_value + regularization
        return total_loss, cross_entropy_loss_value




class FISTA(torch.optim.Optimizer):
    def __init__(self, params, lr, lambda_, nu):
        self.nu = nu
        self.lr = lr
        self.lambda_ = lambda_
        if nu < 1:
            self.phi, self.kappa = self.find_thresh_and_step(lr*lambda_, nu)
        defaults = dict(lr=lr)
        super(FISTA, self).__init__(params, defaults)
    
    def find_thresh_and_step(self, lambda_, nu):
        '''Calculates and returns the threshold and step values.'''
        lambda_cpu = lambda_.cpu().numpy()

        def poly_kappa(kappa):
            return kappa**(2-nu) + 2*kappa + kappa**nu + 2*lambda_cpu*(nu-1)

        sol_kappa = root_scalar(poly_kappa, bracket=[0, lambda_cpu*(1-nu)/2])
        kappa = sol_kappa.root
        phi = kappa / 2 + lambda_cpu / (1 + kappa**(1-nu))
        return float(phi), float(kappa)            

    def nonzerosol(self, u, lambda_):
        '''Calculates the nonzero solution of the new penalty function.'''
        def grad_fun(x, y, lambda_):
            y = np.abs(y)
            return x - y + lambda_ * (1 + self.nu * x**(1-self.nu)) / (1 + x**(1-self.nu))**2

        lambda_cpu = lambda_.cpu().numpy()
        u_cpu = u.cpu().numpy()
        root = newton(grad_fun, self.phi + np.abs(u_cpu), args=(u_cpu, lambda_cpu), maxiter=200, tol=1e-5)
        return root * np.sign(u_cpu)

    def shrinkage_operator(self, u, lambda_):
        '''Applies the shrinkage operator to a PyTorch tensor.'''
        if self.nu < 1:
            sol = torch.zeros_like(u)
            ind = torch.abs(u) >= self.phi
            if not ind.any():
                return sol
            sol[ind] = torch.tensor(self.nonzerosol(u[ind], lambda_), dtype = torch.float, device=u.device)
            return sol
        else:
            return u.sign() * torch.clamp(u.abs() - lambda_, min=0.0)
    
    
    def update(self, new_lr):
        '''Updates the optimizer's learning rate and recalculates phi and kappa.'''
        self.lr = new_lr 
        if self.nu < 1:
            self.phi, self.kappa = self.find_thresh_and_step(new_lr * self.lambda_, self.nu)
        
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {}

    def get_lr(self):
        return self.lr
    
    @torch.no_grad()
    def step(self, closure=None):
        '''Performs a single optimization step.'''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
    
                grad = p.grad.to(p.device)
                state = self.state[p]
    
                if 'x_prev' not in state:
                    state['x_prev'] = p.detach().clone().to(p.device) 
                    state['y_prev'] = p.detach().clone().to(p.device)
                    state['t_prev'] = torch.tensor(1., device=p.device)
    
                x_prev, y_prev, t_prev = state['x_prev'], state['y_prev'], state['t_prev']
    
                x_next = self.shrinkage_operator(y_prev - self.lr * grad, self.lr * self.lambda_)
                t_next = (1. + torch.sqrt(1. + 4. * t_prev ** 2)) / 2.
                y_next = x_next + ((t_prev - 1) / t_next) * (x_next - x_prev)
    
                state['x_prev'].copy_(x_next)
                state['y_prev'].copy_(y_next)
                state['t_prev'].copy_(t_next)
    
                p.data.copy_(x_next)
    
        return loss