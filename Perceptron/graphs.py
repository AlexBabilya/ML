import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from perceptron import Perceptron


class PerceptronGraph:
    def __init__(self, data_path:str, eta: int, n_iter: int) -> None:
        self.data =  pd.read_csv(data_path, header=None, encoding='utf-8')
        
        y = self.data.iloc[:100, 4].values
        self.y = np.where(y == 'Iris-setosa', -1, 1)
        self.X = self.data.iloc[:100, [0, 2]].values
        
        self.perceptron = Perceptron(eta=eta, n_iter=n_iter)
        self.perceptron.fit(self.X, self.y)
    
    def show_data_graph(self) -> None:
        plt.scatter(self.X[:50, 0], self.X[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(self.X[-50:, 0], self.X[-50:, 1], color='blue', marker='x', label='versicolor')
        plt.show()
        
    def show_perceptron_error_graph(self) -> None:
        plt.plot(range(1, len(self.perceptron.errors_) + 1), self.perceptron.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.show()
        
    def _plot_decision_regions(self, resolution):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])
        
        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                             np.arange(x2_min, x2_max, resolution))
        
        Z = self.perceptron.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        for idx, cl in enumerate(np.unique(self.y)):
            plt.scatter(x=self.X[self.y == cl, 0], y=self.X[self.y == cl, 1],
                        alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
            
    def show_decision_regions(self, resolution=0.02):
        self._plot_decision_regions(resolution)
            
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        plt.show()