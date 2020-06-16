#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 05:22:40 2020

@author: melodychen
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import math

ALPHA = 4
test_row_begin = 10 * (ALPHA - 1)
test_row_end = 10 * ALPHA

x_test_1 = []
x_test_2 = []
x_1_train_1 = []
x_1_train_2 = []
x_0_train_1 = []
x_0_train_2 = []


def load_data():
    global y, x, y_test, x_test
    y = []
    x = []
    y_test = []
    x_test = []
    with open("Q2data.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(readCSV):
            if test_row_begin <= index < test_row_end:
                x_test_1.append(float(row[0]))
                x_test_2.append(float(row[1]))
                x_test.append([float(row[0]), float(row[1])])
                y_test.append(float(row[2]))
            else:
                if float(row[2]) == 1:
                    x_1_train_1.append(float(row[0]))
                    x_1_train_2.append(float(row[1]))
                else:
                    x_0_train_1.append(float(row[0]))
                    x_0_train_2.append(float(row[1]))
                x.append([float(row[0]), float(row[1])])
                y.append(float(row[2]))


def plot_data():
    # plot data
    plt.scatter(x_1_train_1, x_1_train_2, label='Label: 1', color='red')
    plt.scatter(x_0_train_1, x_0_train_2, label='Label: 0', color='blue')
    plt.scatter(x_test_1, x_test_2, label='Label: Training', color='cyan')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Plot for KNN Data")
    plt.legend()
    plt.axis([min(x_1_train_1 + x_0_train_1 + x_test_1) - 1, max(x_1_train_1 + x_0_train_1 + x_test_1) + 1,
              min(x_1_train_2 + x_0_train_2 + x_test_2) - 1, max(x_1_train_2 + x_0_train_2 + x_test_2) + 1])
    plt.show()


def find_nearest_neighbor(x_input, actual_output, input_points, k, y_tie):
    # calculate L_1 distance between current point and all training points
    distances = []
    for index, row in enumerate(x):
        l_distance = 0
        for point, input_point in zip(row, input_points):
            l_distance = l_distance + abs(point - input_point)
        distances.append((index, l_distance))
    # store the distance into an array, find the smallest
    sorted_dist = sorted(distances, key=lambda sl: (sl[1], sl[0]))
    output_0 = 0
    output_1 = 0
    for ind in range(k):
        if actual_output[sorted_dist[ind][0]] == 1:
            output_1 = output_1 + 1
        else:
            output_0 = output_0 + 1
    # if there is a tie, return specified y_tie
    if output_0 == output_1:
        return y_tie
    elif output_0 > output_1:
        return 0
    else:
        return 1


def get_testing_error(k, y_tie):
    correct = 0
    for row, output in zip(x_test, y_test):
        if find_nearest_neighbor(x, y, row, k, y_tie) == output:
            correct += 1
    return correct / len(y_test)


def get_training_error(k, y_tie):
    correct = 0
    for row, output in zip(x, y):
        if (find_nearest_neighbor(x, y, row, k, y_tie) == output):
            correct += 1
    return correct / len(y)


def plot(y_tie):
    testing_errors = []
    x_axis = []
    for index in range(1, 10):
        error = get_testing_error(index, y_tie)
        print("K = "+str(index)+" Accuracy = "+str(error))
        testing_errors.append(error)
        x_axis.append(index)
    plt.title('Testing Accuracy for KNN Classifier')
    plt.scatter(x_axis, testing_errors)
    plt.axis([0, 10, 0, 1])
    plt.xlabel('K')
    plt.ylabel('Testing Accuracy')
    plt.show()


def entropy(num, num2):
    x = -num*math.log2(num)-(1-num)*math.log2(1-num)
    x = x*(num2)
    print(x)

def lin_reg():
    x_1 = [0, 4, 5]
    y_1 = [6, 0, 0]
    X = np.array([[1,0],[1,4],[1,5]])
    Y = np.array([[6],[0],[0]])
    w = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
    print(w)
    plt.scatter(x_1, y_1, color='red')
    line_x = np.linspace(min(x_1)-1, max(x_1)+1, 100)
    line_y = float(w[0][0]) + float(w[1][0]) * line_x
    plt.plot(line_x, line_y)
    plt.axis([min(x_1)-1, max(x_1)+1, min(y_1)-1, max(y_1)+1])
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    # load_data()
    # plot(0)
    lin_reg()


