#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 01:19:57 2020

@author: melodychen
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


# load and plot data points
def load_data():
    global x_1_1, x_1_0, x_2_1, x_2_0
    x_1_1 = []
    x_1_0 = []
    x_2_1 = []
    x_2_0 = []
    global x_train, y_train, x
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
    # convert x_train into numpy matrix
    x = np.zeros(shape=(len(x_train), 2))
    for i in range(len(x_train)):
        x[i][0] = x_train[i][0]
        x[i][1] = x_train[i][1]
    # plot all points
    plt.scatter(x_1_1, x_2_1, label='Label: 1', color='red')
    plt.scatter(x_1_0, x_2_0, label='Label: 0', color='orange')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Plot for Data.csv")
    plt.axis([min(x_1_0 + x_1_1) - 1, max(x_1_0 + x_1_1) + 1, min(x_2_1 + x_2_0) - 1, max(x_2_1 + x_2_0) + 1])


# find p(y = 0) from dataset
def find_theta():
    theta = y_train.count(0)/len(y_train)
    return theta


# find mu0 or mu1 from dataset
def find_mu(num):
    # calculate number of occurences of num
    n = y_train.count(num)
    # add up all vectors with y = num
    mu = np.zeros(shape=(1, len(x[0])))
    for index in range(len(x)):
        if y_train[index] == num:
            mu += x[index]
    # divide mu by n
    mu = mu / n
    return mu


# find sigma, aka covariance matrix
def find_sigma():
    n1 = y_train.count(1)
    n0 = y_train.count(0)
    s1 = np.zeros(shape=(len(x[0]), len(x[0])))
    s0 = np.zeros(shape=(len(x[0]), len(x[0])))
    for index in range(len(x)):
        if y_train[index] == 1:
            s1 += np.matmul(np.subtract(x[index], find_mu(1)).transpose(), np.subtract(x[index], find_mu(1)))
        else:
            s0 += np.matmul(np.subtract(x[index], find_mu(0)).transpose(), np.subtract(x[index], find_mu(0)))
    s1 = s1 / n1
    s0 = s0 / n0
    sigma = s1 * (n1/(n1 + n0)) + s0 * (n0/(n1 + n0))
    return sigma


# finds w and b, and plots contour
def find_decision_boundary():
    mu0 = find_mu(0)
    mu1 = find_mu(1)
    sigma = find_sigma()
    sigma_inv = np.linalg.inv(sigma)
    # uses formula derived in hw for w and b
    w = np.matmul(sigma_inv, (mu0-mu1).transpose())
    b = -0.5*np.matmul(np.matmul(mu0, sigma_inv), mu0.transpose()) + 0.5*np.matmul(np.matmul(mu1, sigma_inv), mu1.transpose()) + np.log(find_theta()/(1-find_theta()))
    print("w: ")
    print(str(w))
    print("b: " + str(b[0][0]))
    # plot line perpendicular to vector w, decision boundary
    x_line = np.linspace(min(x_1_0 + x_1_1) - 1, max(x_1_0 + x_1_1) + 1, 100)
    # equation of line, we know that w1*x1 + w2*x2 + b = 0, x2 = (-w1*x1)/w2 - b/w2
    y_line = (-(float(w[0][0]) / float(w[1][0])) * x_line - (float(b[0][0]) / float(w[1][0])))
    plt.plot(x_line, y_line, color='blue', label='Decision Boundary')
    # plotting the two contours
    delta = 0.025
    x = np.arange(-3.0, 11.0, delta)
    y = np.arange(-7.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)  # generates coordinates for contour plots
    # 2D array for value at each contour plot coordinates
    Z = [[0 for x1 in range(len(X[0]))] for y1 in range(len(X))]
    Z1 = [[0 for x1 in range(len(X[0]))] for y1 in range(len(X))]
    # calculates p(x,y) for each pair of coordinates
    for i in range(len(X)):
        for j in range(len(X[0])):
            # uses formula p(x,y) = p(x|y)*p(y), p(x|y) is gaussian distribution
            Z[i][j] = (1/(2*np.pi*(np.linalg.det(sigma)**0.5)))*np.exp(-0.5*(np.array([X[i][j]-mu0[0][0], Y[i][j]-mu0[0][1]]).transpose())@sigma_inv@np.array([X[i][j]-mu0[0][0], Y[i][j]-mu0[0][1]]))*find_theta()
            Z1[i][j] = (1/(2*np.pi*(np.linalg.det(sigma)**0.5)))*np.exp(-0.5*(np.array([X[i][j]-mu1[0][0], Y[i][j]-mu1[0][1]]).transpose())@sigma_inv@np.array([X[i][j]-mu1[0][0], Y[i][j]-mu1[0][1]]))*(1-find_theta())
    plt.contour(X, Y, Z, levels=np.logspace(-3, -1, 7), colors='c')  # Contour for P(x, y = 0)
    plt.contour(X, Y, Z1, levels=np.logspace(-3, -1, 7), colors='m')  # Contour for P(x, y = 1)


if __name__ == "__main__":
    # part a, load and plot data
    load_data()
    # part b
    print("p(y = 0): " + str(find_theta()))
    print("mu 0: ")
    print(str(find_mu(0).transpose()))
    print("mu 1: ")
    print(str(find_mu(1).transpose()))
    print("sigma: ")
    print(str(find_sigma()))
    # part c and d
    find_decision_boundary()
    # show legend
    plt.legend(loc='upper left')
    # plot the graph
    plt.show()

