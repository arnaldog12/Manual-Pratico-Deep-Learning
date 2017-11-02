import numpy as np
import matplotlib.pyplot as plt

def plot_nn_classifier_3d_in_2d(x, y, nn, threshold=0.0, figsize=(6,6), s=100, cmap='bwr'):
    x1, x2 = np.meshgrid(np.linspace(x[:,0].min(), x[:,0].max(), 100), np.linspace(x[:,1].min(), x[:,1].max(), 100))
    x_mesh = np.array([x1.ravel(), x2.ravel()]).T
    y_mesh = nn.predict(x_mesh)
    y_mesh = np.where(y_mesh <= threshold, 0, 1)
    
    plt.figure(0, figsize=figsize)
    plt.scatter(x[:,0], x[:,1], c=list(np.array(y).ravel()), s=s, cmap=cmap)
    plt.contourf(x1, x2, y_mesh.reshape(x1.shape), cmap=cmap, alpha=0.5)
    
    plt.show()