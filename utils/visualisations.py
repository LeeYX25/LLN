import matplotlib.pyplot as plt
import numpy as np
from seaborn import color_palette
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from IPython.display import HTML
import torch
from bisect import bisect_right

def lasso_path(layer1_history, imp_feat):
    palette = color_palette(None, len(imp_feat))

    plt.figure(figsize=(15,6))
    start_x = 0
    x_ticks, lambda_values = [], []

    for lambi, parameter_values in layer1_history.items():
        x_ticks.append(start_x)
        lambda_values.append(np.round(lambi, 3))
        plt.axvline(x=start_x, color='gray', linestyle='--')

        tensors = [torch.norm(value, p=2, dim=0) for value in parameter_values]
        data = torch.stack(tensors, dim=0)
        x_values = np.arange(start_x, start_x + data.size(0))

        for k in range(data.size(1)):
            color_index = imp_feat.index(k) if k in imp_feat else None
            color = palette[color_index] if color_index is not None else 'blue'
            linestyle = 'solid' if k in imp_feat else 'dashed'
            linewidth = 2 if k in imp_feat else 0.5
            label = f'{k}' if lambi == list(layer1_history.keys())[-1] and k in imp_feat else None
            plt.plot(x_values, data[:, k].numpy(), color=color, linestyle=linestyle, linewidth=linewidth, label=label)

        start_x += data.size(0)

            
    plt.legend(loc='lower right')
    plt.title("Lasso Path")
    plt.xlabel("Regularisation strength")
    plt.ylabel("Parameters magnitudes")
    plt.xticks(x_ticks, lambda_values)
    plt.show()


def draw_layer1_evolution(layer1_history):

    def check_imp_feat_changes(weight, previous_important_features):
        non_zero_columns = torch.any(weight != 0, dim=0)
        indices = torch.where(non_zero_columns)[0]
        new_important_features = set(indices.tolist())
        if new_important_features != previous_important_features:
            return True, new_important_features
        return False, previous_important_features


    all_weights, lambdas, epochs = [], [], []

    for lambi, weights in layer1_history.items():
        lambdas.append(lambi)
        current_key_weights = []
        previous_important_features = set()

        for weight in weights:
            changed, new_features = check_imp_feat_changes(weight, previous_important_features)
            if changed:
                current_key_weights.append(weight)
            previous_important_features = new_features  # Mettre à jour les caractéristiques pour la prochaine comparaison

        all_weights.append(current_key_weights)
        epochs.append(len(current_key_weights))


    epochs = np.array(epochs)
    lim_weights = all_weights[0][0].shape

    fig, ax = plt.subplots(1, figsize=(10, 5))
    im = ax.imshow(all_weights[0][0], aspect=lim_weights[1]/lim_weights[0], extent=[0,lim_weights[1],0,lim_weights[0]], cmap='inferno', origin='lower')

    def find_index(i, lambdas, epochs):
        epochs_sum = np.cumsum(epochs)
        index = bisect_right(epochs_sum, i)
        if index != 0:
            return index, i - epochs_sum[index-1]
        return index, i

    def init():
        im.set_data(all_weights[0][0])
        return im

    def update(i):
        index, limit = find_index(i, lambdas, epochs)
        im.set_data(all_weights[index][limit])
        fig.suptitle(f"Training Run: λ={lambdas[index]:.2f}")
        return im

    anim = FuncAnimation(fig, update, frames=range(np.cumsum(epochs)[-1]), init_func=init)

    def is_running_in_notebook():
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:
                return False
        except Exception:
            return False
    
        return True

    if is_running_in_notebook():
        return HTML(anim.to_jshtml())
    else:
        return plt.show()