#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:19:57 2020

@author: melodychen
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import cvxpy as cp

# global variables
x_train = []
y_train = []
x_1_1 = []
x_1_0 = []
x_2_1 = []
x_2_0 = []


def load_data():
    global x_1_1, x_1_0, x_2_1, x_2_0
    x_1_1 = []
    x_1_0 = []
    x_2_1 = []
    x_2_0 = []
    global x_train, y_train
    x_train = []
    y_train = []
    with open("data.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if int(row[2]) == 1:
                x_1_1.append(float(row[0]))
                x_2_1.append(float(row[1]))
            else:
                x_1_0.append(float(row[0]))
                x_2_0.append(float(row[1]))
            x_train.append([float(row[0]), float(row[1])])
            y_train.append(float(row[2]))
    # plot all points
    plt.scatter(x_1_1, x_2_1, label='Label: 1', color='red')
    plt.scatter(x_1_0, x_2_0, label='Label: -1', color='orange')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("SVM Plot for Data.csv")
    plt.axis([min(x_1_0 + x_1_1) - 1, max(x_1_0 + x_1_1) + 1, min(x_2_1 + x_2_0) - 1, max(x_2_1 + x_2_0) + 1])


def primal_problem():
    # load data
    x = np.zeros(shape=(len(x_train), 2))
    y = np.zeros(shape=(len(x_train), 1))
    # load into numpy array
    for index, row in enumerate(x_train):
        x[index][0] = row[0]
        x[index][1] = row[1]
        y[index] = y_train[index]
    # variables we're using to minimize
    w = cp.Variable(2)
    b = cp.Variable(1)
    # function we're trying to minimize
    cost = cp.sum_squares(w)
    # constraints for minimization
    constraints = []
    for index, row in enumerate(x):
        constraints.append(y[index] * (w.T @ row + b) >= 1)
    # use cvxpy to do minimization
    prob = cp.Problem(cp.Minimize(cost), constraints)
    result = prob.solve()
    # final values
    print("w vector: "+str(w.value))
    print("b: "+str(b.value))
    # plot line perpendicular to vector w, decision boundary
    x_line = np.linspace(min(x_1_0 + x_1_1)-1, max(x_1_0 + x_1_1)+1, 100)
    # equation of line, we know that w1*x1 + w2*x2 + b = 0, x2 = (-w1*x1)/w2 - b/w2
    y_line = (-(float(w.value[0])/float(w.value[1]))*x_line-(float(b.value)/float(w.value[1])))
    plt.plot(x_line, y_line, color='blue', label='Decision Boundary')


def dual_problem():
    # load data
    x = np.zeros(shape=(len(x_train), 2))
    y = np.zeros(shape=(len(x_train), 1))
    for index, row in enumerate(x_train):
        x[index][0] = row[0]
        x[index][1] = row[1]
        y[index] = y_train[index]
    # variable used to maximize dual
    alpha = cp.Variable(len(x_train))
    # P matrix that represents part of latter part of W(a)
    p = np.zeros(shape=(len(x_train), len(x_train)))
    # Fill up P matrix with y_i*y_j*x_i^T*x_j
    for r in range(len(x)):
        for c in range(len(x)):
            p[r][c] = y[r]*y[c]*x[r].transpose().dot(x[c])
    # slight adjustment to P
    p = p + 1e-13 * np.eye(31)
    # function we're trying to maximize
    cost = sum(alpha) - 0.5 * cp.quad_form(alpha, p)
    # our constraints
    constraints = []
    for a in alpha:
        constraints.append(0 <= a)
    constraints.append(sum(alpha * y) == 0)
    # using cvxpy to solve maximization problem
    prob = cp.Problem(cp.Maximize(cost), constraints)
    result = prob.solve()
    print("Original Alphas: ")
    print(alpha.value)
    # we want to make very small values zero
    alpha_float = []
    non_zero_alpha = []
    for index, num in enumerate(alpha.value):
        if float(num) < 1e-9:
            alpha_float.append(0)
        else:
            alpha_float.append(float(num))
            non_zero_alpha.append((index, float(num)))
    print("Cleaned up version of Alphas: ")
    print(alpha_float)
    print(non_zero_alpha)
    # want to highlight these points
    x_1_highlight = []
    x_2_highlight = []
    for tup in non_zero_alpha:
        x_1_highlight.append(x[tup[0]][0])
        x_2_highlight.append(x[tup[0]][1])
    # highlights support vector in plot
    plt.scatter(x_1_highlight, x_2_highlight, color='purple', label='Label: Support Vector')


if __name__ == "__main__":
    load_data()  # part a
    primal_problem()  # part b
    dual_problem()  # part c
    # show legend
    plt.legend(loc='upper right')
    # plot the graph
    plt.show()
