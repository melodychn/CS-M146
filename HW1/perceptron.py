#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:34:15 2020

@author: melodychen
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import math

def plot_data(set_num):
    x_1 = []
    x_2 = []
    y_1 = []
    y_2 = []
    with open('data'+str(set_num)+'.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(int(row[2]) == 1):
                x_1.append(float(row[0]))
                y_1.append(float(row[1]))
            else:
                x_2.append(float(row[0]))
                y_2.append(float(row[1]))
    plt.scatter(x_1, y_1, color='red')
    plt.scatter(x_2, y_2)
    plt.axis([min(x_1+x_2), max(x_1+x_2), min(y_1+y_2), max(y_1+y_2)])
    plt.show()
def plot_all_data():
    plot_data(1)
    plot_data(2)
    plot_data(3)
def plot(w, set_num):
    x_1 = []
    x_2 = []
    y_1 = []
    y_2 = []
    with open('data'+str(set_num)+'.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(int(row[2]) == 1):
                x_1.append(float(row[0]))
                y_1.append(float(row[1]))
            else:
                x_2.append(float(row[0]))
                y_2.append(float(row[1]))
    plt.scatter(x_1, y_1, color='red')
    plt.scatter(x_2, y_2)
    x = np.linspace(min(x_1+x_2), max(x_1+x_2), 100)
    y = (-w[1]*x-w[0])/w[2]
    plt.plot(x, y)
    plt.axis([min(x_1+x_2), max(x_1+x_2), min(y_1+y_2), max(y_1+y_2)])
    plt.show()

def perceptron(maxIter, set_num):
    #create vectors from points
    points = []
    with open('data'+str(set_num)+'.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            temp = [1]
            temp.append(float(row[0]))
            temp.append(float(row[1]))
            temp.append(int(row[2]))
            points.append(temp)
    #initialize
    w = [0, 0, 0]
    iters = 0
    u = 0 #number of updates
    #evaluate at each point
    while(iters < maxIter):
        for vector in points:
            #dot product of vector and w
            product = vector[0]*w[0] + vector[1]*w[1] + vector[2]*w[2]
            if (product*vector[3] <= 0):
                u+=1
                x = w[0]+vector[3]*vector[0]
                y = w[1]+vector[3]*vector[1]
                z = w[2]+vector[3]*vector[2]
                w = [x,y,z]
        iters+=1
    print('w vector: '+str(w))
    print('w1: '+str(w[1])+' w2: '+str(w[2]))
    print('b bias: '+str(w[0]))
    print('# of updates: '+str(u))
    compute_margin(w, points)
    plot(w, set_num)

def compute_margin(w, points):
    margin = []
    for vector in points:
        product = vector[0]*w[0] + vector[1]*w[1] + vector[2]*w[2]
        mag = math.sqrt(w[1]*w[1]+w[2]*w[2])
        margin.append(abs(product)/mag)
    gamma = min(margin)
    print("Margin: "+str(gamma))
    print("Upper Bound: "+str(1/(gamma*gamma)))

if __name__ == "__main__":
    #plot_all_data()
    perceptron(1000, 2)