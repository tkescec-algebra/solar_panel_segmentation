import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_losses(train_losses, val_losses, model_name):
    sns.lineplot(x=range(len(train_losses)), y=train_losses).set(title='Train Loss')
    plt.savefig(f'results/plots/{model_name}_train_loss.png')
    plt.show()

    sns.lineplot(x=range(len(train_losses)), y=val_losses).set(title='Validation Loss')
    plt.savefig(f'results/plots/{model_name}_val_loss.png')
    plt.show()

def plot_metrics(metrics, title, x_label, y_label, legend=None):
    plt.figure(figsize=(10, 6))
    for item in metrics:
        plt.plot(item['data'], label=item['name'])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    img_name = title.replace(' ', '_').lower()
    plt.savefig(f'results/plots/{img_name}.png')
    plt.show()