import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Using the TkAgg backend for the matplotlib
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("MacOSX")

import numpy as np

from neurons.dataset import get_dataset

# Select Target Variable (Sentosa and Versicolor Flowers) from the Iris dataset and mark them as 0 and 1
dataset = get_dataset("IRIS")
target = dataset.iloc[0:100, 4].values
target =  np.where(target == 'Iris-setosa', 0, 1)

# Select the first two features (Sepal Length and Petal Length) for training
features = dataset.iloc[0:100, [0, 2]].values

# Create a Scatter Plot for the data
plt.scatter(features[:50, 0], features[:50, 1], color='red', marker='o', label='Sentosa')
plt.scatter(features[50:100, 0], features[50:100, 1], color='blue', marker='x', label='Versicolor')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(loc='upper left')
plt.show()
