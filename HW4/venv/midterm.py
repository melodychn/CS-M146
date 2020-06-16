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


def load_data():
    global y, x, y_test, x_test
    y = []
    with open("dataTraining_Y.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            y.append(float(row[0]))
    x = []
    with open("dataTraining_X.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
    y_test = []
    with open("dataTesting_Y.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            y_test.append(float(row[0]))
    x_test = []
    with open("dataTesting_X.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x_test.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])


def find_nearest_neighbor(x_input, actual_output, input_points, k, y_tie):
    # calculate eucledean distance between current point and all training points
    distances = []
    for index, row in enumerate(x):
        euc_distance = 0
        for point, input_point in zip(row, input_points):
            euc_distance = euc_distance + ((point - input_point) ** 2)
        euc_distance = euc_distance ** 0.5
        distances.append((index, euc_distance))
    # store the distance into an array, find the smallest
    sorted_dist = sorted(distances, key=lambda sl: (sl[1], sl[0]))
    output_0 = 0
    output_1 = 0
    for ind in range(k):
        if (actual_output[sorted_dist[ind][0]] == 1):
            output_1 = output_1 + 1
        else:
            output_0 = output_0 + 1

    if (output_0 == output_1):
        return y_tie
    elif (output_0 > output_1):
        return 0
    else:
        return 1


def get_testing_error(k, y_tie):
    correct = 0
    for row, output in zip(x_test, y_test):
        if (find_nearest_neighbor(x, y, row, k, y_tie) == output):
            correct += 1
    return correct / len(y_test)


def get_training_error(k, y_tie):
    correct = 0
    for row, output in zip(x, y):
        if (find_nearest_neighbor(x, y, row, k, y_tie) == output):
            correct += 1
    return correct / len(y)


def plot(y_tie):
    training_errors = []
    testing_errors = []
    x_axis = []
    for index in range(1, 16):
        training_errors.append(1 - get_training_error(index, y_tie))
        testing_errors.append(1 - get_testing_error(index, y_tie))
        x_axis.append(index)
    plt.title('Y_tie = ' + str(y_tie))
    plt.scatter(x_axis, training_errors, color='red', label='training error')
    plt.scatter(x_axis, testing_errors, label='testing error')
    plt.axis([0, 16, 0, 0.33])
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    load_data()
    # plot for y_tie = 1
    plot(1)
    plot(0)
