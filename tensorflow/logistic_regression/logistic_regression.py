#!/usr/bin/env python
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()

# iris.data looks like [[5.1  3.5  1.4  0.2],...] where the quadruples are the
# lengths and the widths of the sepals and petals, in centimetres
print("iris.data--------------------")
print(iris.data)
print "iris.data length:", len(iris.data)

# iris.target looks like [0, 1, 2,...] where the numbers indicate the types of
# Iris, 0, 1, 2 for Iris setosa, Iris virginica and Iris versicolor, resp.
print("iris.target------------------")
print(iris.target)
print "iris.target length:", len(iris.target)

iris_X, iris_y = iris.data[:-1, :], iris.target[:-1]
print "iris_X length:", len(iris_X)
print "iris_y length:", len(iris_y)

iris_y = pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(
    iris_X, iris_y, test_size=0.33, random_state=42)
