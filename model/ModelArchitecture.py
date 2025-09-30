import torch 
import torch.nn as nn
import utils
import numpy as np
import pandas as pd
from torch.nn.functional import normalize, linear
from skgarden import MondrianForestRegressor,RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from collections import Counter
import math
import torch.nn.functional as F

class ModelArchitecture:
    def __init__(self, penalty=0, model_type=0, p2=20, lambda_qut=None, device=None):
        self.penalty = penalty # 0 for custom and 1 for l1
        self.model_type = model_type # 0 for regression and 1 for classification
        self.p2 = p2
        self.lambda_qut = lambda_qut
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') if device is None else device
        self.act_fun = utils.Custom_act_fun()

        # Attributes to be modified after training
        self.trained = False
        self.layer1, self.layer2 = None, None
        self.important_features = None
        self.layer1_simplified = None
        self.layer1_history = {}

        
    def rf_graph(self,x):
            
        G = np.zeros((len(x), len(x)))
        
        for i in range(len(x)):
            
            G[i, np.where(x == x[i])[0]] = 1

        nodes = Counter(x)
        nodes_num = np.array([nodes[i] for i in x])
            
        return G, G / nodes_num.reshape(-1, 1)

    
    def get_rfweight(self,rf, x):
        
        n = x.shape[0]
        
        leaf = rf.apply(x)
        ntrees = leaf.shape[1]
        G_unnorm = np.zeros((n, n))
        G_norm = np.zeros((n, n))
        
        for i in range(ntrees):
            
            tmp1, tmp2 = self.rf_graph(leaf[:, i])
            G_unnorm += tmp1
            G_norm += tmp2
        
        return G_unnorm / ntrees, G_norm / ntrees
    

    def get_rfmatrix(self,x,y):
        params_mrf = {'min_samples_split': [2, 3, 4, 5, 6, 7]}
        # if y.shape[1]>1:
        #     model = RandomForestRegressor(n_estimators=100)
        # else:
        model = MondrianForestRegressor(n_estimators=100)
            
        
        reg_mrf = GridSearchCV(model, params_mrf)
        n=x.shape[0]
        reg_mrf.fit(x, y)
        mrf = reg_mrf.best_estimator_
        mrf.fit(x, y)
        mrfw,mrfwn= self.get_rfweight(mrf, x)


        return mrfw,mrfwn
    
    def classloss2(self,y,fx,w):
        n=fx.shape[0]
        m=fx.shape[1]
        log_softmax = F.log_softmax(fx, dim=1)
        y = torch.tensor(y, dtype=torch.long)
        Y=torch.nn.functional.one_hot(y, num_classes=m).float().squeeze(1).to('cuda')
        Y_log_mu = torch.matmul(Y, log_softmax.T) 
        loss = -torch.sum(Y_log_mu * w)
        loss=loss/(n*(n-1))
        return(loss)
    
    def fit(self, X, y, tau,verbose=False, param_history=False,vstat=False,norm=False):
        self.norm=norm
        self.tau = tau
        self.X_data=X
        self.y_data =y
        self.loss2=None
        if (vstat==True):
            self.tau= 1/(X.shape[0])

        if self.trained:
            raise Exception("Model already trained, call 'reset' method first.")
        #py=y.shape[1]
        rf_weight = self.get_rfmatrix(self.X_data, self.y_data)
        
        
        
        if self.norm==False:
            self.weight=rf_weight[0]
        else:
            self.weight = rf_weight[1]


        sum_w=np.sum(self.weight)
        
        n=X.shape[0]
        self.batch_size=n
        
        if sum_w<=n:
            print("w is an identity matrix")
            self.tau=1
            
        self.weight = torch.FloatTensor(self.weight).to(self.device)
        
        X, y = utils.data_to_tensor(X, y)
        
        if self.model_type == 1:
            y = y.long()
            hat_p = utils.get_hat_p(y)

        X, y = X.to(self.device), y.to(self.device)

        n_features, p1 = X.shape

        output_dim = 1 if self.model_type == 0 else len(hat_p)
        self.layer1 = nn.Linear(p1, self.p2, dtype=torch.float, device=self.device)
        self.layer2 = nn.Linear(self.p2, output_dim, dtype=torch.float, device=self.device)
        nn.init.normal_(self.layer1.weight, mean=0.0, std=10) #! std = y.std() ??!!
        nn.init.normal_(self.layer2.weight, mean=0.0, std=10)
        #self.lambda_qut=31.2060
        if self.lambda_qut is None:
            if self.model_type == 0:
                self.lambda_qut = utils.lambda_qut_regression(X, self.act_fun,self.tau,self.weight)
            elif self.model_type == 1:
                self.lambda_qut = utils.lambda_qut_classification(X, hat_p, self.act_fun,self.tau,self.weight)
        else:
            if isinstance(self.lambda_qut, torch.Tensor):
                self.lambda_qut = self.lambda_qut.to(self.device, dtype=torch.float)
            else:
                self.lambda_qut = torch.tensor(self.lambda_qut, dtype=torch.float, device=self.device)
        print(self.lambda_qut)
        self.train_loop(X, y, verbose, param_history)

        
        # Check if returning mean(y) gives better results
        if self.model_type == 0 :
            if self.penalty == 0:
                loss_fn = utils.CustomRegressionLoss(self.lambda_qut, 0.1).to(self.device)
            else:
                loss_fn = utils.CustomRegressionLoss(self.lambda_qut, 1).to(self.device)
            model_error = torch.sqrt((self.tau*loss_fn(self.forward(X).flatten(), y, self.layer1)[1])**2+(1-self.tau)*self.loss2)+(loss_fn(self.forward(X).flatten(),y,self.layer1)[0]-loss_fn(self.forward(X).flatten(), y,self.layer1)[1])
            
            
            metric = torch.nn.MSELoss(reduction='sum')
            baseline_error = torch.sqrt(metric(y.mean(), y))

            if model_error > baseline_error:
                self.layer1.weight.data.fill_(0)
                self.layer1.bias.data.fill_(0)
                self.layer2.weight.data.fill_(0)
                self.layer2.bias.data.fill_(y.mean().item())
                print("choose baseline model")
                
        self.important_features = self.imp_feat()
        self.layer1_simplified = nn.Linear(self.important_features[0], self.p2, device=self.device)
        self.layer1_simplified.weight.data, self.layer1_simplified.bias.data = self.layer1.weight.data[:, self.important_features[1]].clone(), self.layer1.bias.data.clone()

        if verbose: print("MODEL FITTED !")
        self.trained = True

    def train_loop(self, X, y,verbose, param_history):

        for i in range(-1, 6):
            nu = 1 if self.penalty == 1 else [1, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1][i + 1]
            lambi = self.lambda_qut * (np.exp(i) / (1 + np.exp(i)) if i < 5 else 1)

            rel_err = 1e-9 if i == 5 else 1e-5
            ab_err= 1e-5

            loss_fn = utils.CustomRegressionLoss(lambi, nu).to(self.device) if self.model_type == 0 else utils.CustomClassificationLoss(lambi, nu).to(self.device)
            init_lr = 0.1*0.8**(i+1)

            if verbose:
                print(f"Lambda = {lambi.item():.4f} -- Nu = {nu}" if self.penalty == 0 else f"Lambda = {lambi.item():.4f}")
            
            self.train(nu, X, y,lambi, init_lr, rel_err, loss_fn, verbose, param_history)

    def train(self, nu, X, y,lambda_, initial_lr, rel_err, loss_fn, verbose, param_history):
        layer1, layer2 = self.layer1, self.layer2
        epoch, last_loss = 0, np.inf
        optimizer_l1 = utils.FISTA(params=layer1.parameters(), lr=initial_lr, lambda_=lambda_, nu=nu)
        optimizer_l2 = utils.FISTA(params=layer2.parameters(), lr=initial_lr, lambda_=torch.tensor(0, dtype=torch.float, device = self.device), nu=nu)

        lr_factor = 0.9
        max_epochs = 10000

        X_rf = torch.FloatTensor(self.X_data.astype(float))
        y_rf= torch.FloatTensor(self.y_data.astype(float))
        n = X.shape[0]
        p = X.shape[1]
        rf_weight=self.weight
        w = rf_weight if self.tau is None else rf_weight - torch.diag(torch.diag(rf_weight))
        #yp=self.y_data.shape[1]
        if param_history: self.layer1_history[lambda_.item()] = [layer1.weight.data.clone()]

        while epoch < max_epochs:

            if (self.model_type == 0):
                y_pred = self.forward(X)
                y_pred1= y_pred.view(self.batch_size,1)
                # y_m = torch.from_numpy(self.y_data).to(self.device)
                # y_m=y_m.expand(-1, yp) 
                # y_m = torch.tile(y_m.unsqueeze(0), (self.batch_size, 1,1)).to(self.device)
                # y_predm = torch.tile(y_pred.unsqueeze(1), (1, self.batch_size,1)).to(self.device)
                y_m = torch.tile(y, (self.batch_size, 1))
                y_predm= torch.tile(y_pred1, (1, self.batch_size))
                
                
                #loss2 = torch.mean((y_m - y_predm ) ** 2 * w.unsqueeze(-1))*n*n
                loss2 = torch.sum((y_m - y_predm ) ** 2 * w)/(n*(n-1))
                self.loss2=loss2
                optimizer_l1.zero_grad()
                optimizer_l2.zero_grad()
                loss=torch.sqrt(self.tau*loss_fn(y_pred.flatten(), y, layer1)[1]**2/n+(1-self.tau)*loss2)+loss_fn(y_pred.flatten(), y, layer1)[0]-loss_fn(y_pred.flatten(), y, layer1)[1]
                bare_loss = torch.sqrt(self.tau*loss_fn(y_pred.flatten(), y, layer1)[1]**2/n+(1-self.tau)*loss2)         

            elif(self.model_type == 1):
                y_pred= self.forward(X)
                loss2=self.classloss2(y,y_pred,w)
                self.loss2=loss2
                optimizer_l1.zero_grad()
                optimizer_l2.zero_grad()
                loss=self.tau*loss_fn(y_pred, y, layer1)[1]+(1-self.tau)*loss2+(loss_fn(y_pred, y, layer1)[0]-loss_fn(y_pred, y, layer1)[1])
                bare_loss = self.tau*loss_fn(y_pred, y, layer1)[1]+(1-self.tau)*loss2
                
                
            loss = loss.detach()

            if verbose and epoch % 20 ==0:
                print(f"\tEpoch: {epoch} | Loss: {loss.item():.5f} | learning rate : {optimizer_l1.get_lr():.6f}")

            if loss > last_loss: 
                optimizer_l1.update(optimizer_l1.get_lr()*lr_factor)
                optimizer_l2.update(optimizer_l2.get_lr()*lr_factor)
            if epoch % 10 == 0:
                if epoch > 0 and abs(loss - last_loss) / loss < rel_err:
                #if epoch > 0 and abs(loss - last_loss)  < ab_err:
                    if verbose: print(f"\n\t Descent stopped: loss is no longer decreasing significantly.\n")
                    break
                last_loss = loss
                
            epoch += 1
            bare_loss.backward(retain_graph=True)
            optimizer_l1.step()
            optimizer_l2.step()

            if param_history: self.layer1_history[lambda_.item()].append(layer1.weight.data.clone())

        if epoch == max_epochs and verbose: print("FISTA descent stopped: maximum iterations reached") 


    def reset(self):
        self.trained = False
        self.important_features = None
        self.layer1_simplified = None
        self.layer1, self.layer2 = None, None
        
    def imp_feat(self):
        weight = self.layer1.weight.data
        non_zero_columns = torch.any(weight != 0, dim=0)
        indices = torch.where(non_zero_columns)[0]
        count = torch.sum(non_zero_columns).item()
        return count, sorted(indices.tolist())

    def apply_feature_selection(self, X):
        input_type = type(X)
        X_tensor = utils.X_to_tensor(X).to(self.device)
        X_selected = X_tensor[:, self.important_features[1]]

        if input_type == pd.DataFrame:
            return pd.DataFrame(X_selected.cpu().numpy(), columns=[X.columns[i] for i in self.important_features[1]])
        if input_type == torch.Tensor:
            return X_selected
        else:
            return X_selected.cpu().numpy()

    def fit_and_apply(self, X, y, verbose=False):
        self.fit(X, y, verbose)
        X = self.apply_feature_selection(X)
        if verbose: print("Features selection applied !")
        return X
            
    def predict(self, X):
        if not self.trained:
            raise Exception("Model not trained, call 'fit' method first.")

        X = utils.X_to_tensor(X).to(self.device)
        if (self.important_features[0]>0):
            X = self.apply_feature_selection(X)
            with torch.inference_mode():
                layer1_output = self.act_fun(self.layer1_simplified(X))
                w2_weights_normalized = normalize(self.layer2.weight, p=2, dim=1)
                logits = linear(layer1_output, w2_weights_normalized, self.layer2.bias)

        elif(self.important_features[0]==0):
            with torch.inference_mode():
                layer1_output = self.act_fun(self.layer1(X))
                w2_weights_normalized = normalize(self.layer2.weight, p=2, dim=1)
                logits = linear(layer1_output, w2_weights_normalized, self.layer2.bias)
                
        if self.model_type == 0:
            return logits.squeeze().cpu().numpy() # Regression

        softmax = nn.Softmax(dim=1)
        return torch.argmax(softmax(logits), dim=1).cpu().numpy() # Classification

    def lasso_path(self):
        if not self.trained:
            raise Exception("Model not trained, call 'fit' method first and set 'param_history' to True.")
        if self.trained and not bool(self.layer1_history):
            raise Exception("Model has been trained with 'param_history' set to False.")

        return utils.lasso_path(self.layer1_history, self.important_features[1])
    
    def layer1_evolution(self):
        if not self.trained:
            raise Exception("Model not trained, call 'fit' method first and set 'param_history' to True.")
        if self.trained and not bool(self.layer1_history):
            raise Exception("Model has been trained with 'param_history' set to False.")

        return utils.draw_layer1_evolution(self.layer1_history)

    def info(self):
        print("MODEL INFORMATIONS:")
        print('=' * 20)
        print("General:")
        print('―' * 20)
        print(f"  Training Status: {'Trained' if self.trained else 'Not Trained'}")
        if self.trained:
            print(f"  Lambda_qut: {np.round(self.lambda_qut.item(), 4)}\n")
            print("Layers:")
            print('―' * 20)
            print("  Layer 1: ")
            print(f"\t Shape = {list(self.layer1.weight.shape)}")
            print(f"\t Number of non zero entries in weights: {self.important_features[0]}")
            print(f"\t Non zero entries indexes: {self.important_features[1]}")
            print(f"\t Call 'layer1_simplified' attribute to get full, non zero, first layer.")
            print("  Layer 2:")
            print(f"\t Shape = {list(self.layer2.weight.shape)}")
            
    def acc(self,y,ypred):
        equal_count = np.sum(y == ypred)
        n=y.shape[0]
        acc=equal_count/n
        return(acc)
        
    




