import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from graphs import PerceptronGraph

s = os.path.join('https://archive.ics.uci.edu', 
                 'ml', 'machine-learning-databases', 
                 'iris','iris.data'
                )
graph = PerceptronGraph(data_path=s, eta=0.1, n_iter=10)

graph.show_data_graph()
graph.show_perceptron_error_graph()
graph.show_decision_regions()