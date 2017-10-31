import numpy as np
import matplotlib.pyplot as plt

def plot_linear_classifier_2d_nn(x, y, nn, threshold=0.0, figsize=(6,6), s=100, cmap='bwr'):
    x1, x2 = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    x_mesh = np.array([x1.ravel(), x2.ravel()]).T
    y_mesh = nn.predict(x_mesh)
    y_mesh = np.where(y_mesh <= threshold, 0, 1)
    
    plt.figure(0, figsize=figsize)
    plt.scatter(x[:,0], x[:,1], c=list(y), s=s, cmap=cmap)
    plt.contourf(x1, x2, y_mesh.reshape(x1.shape), cmap=cmap)
    
    plt.show()