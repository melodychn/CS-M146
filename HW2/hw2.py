#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:09:29 2020

@author: melodychen
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import math
x_1 = []
x_1_test = []
y_1 = []
y_1_test = []

def load_data():
    #load train data
    with open('regression_train.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x_1.append(float(row[0]))
            y_1.append(float(row[1]))
    global x, y, x_test, y_test
    x = np.zeros(shape=(len(x_1), 2))
    y = np.zeros(shape=(len(x_1), 1))
    for i in range(len(x_1)):
        x[i][0] = 1
        x[i][1] = x_1[i]
        y[i] = y_1[i]
    with open('regression_test.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x_1_test.append(float(row[0]))
            y_1_test.append(float(row[1]))
    x_test = np.zeros(shape=(len(x_1_test), 2))
    y_test = np.zeros(shape=(len(x_1_test), 1))
    for i in range(len(x_1_test)):
        x_test[i][0] = 1
        x_test[i][1] = x_1_test[i]
        y_test[i] = y_1_test[i]

def plot():
    plt.scatter(x_1, y_1, color='red')
    plt.axis([min(x_1), max(x_1), min(y_1), max(y_1)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def linear_regression_closed_form():
    x_t = x.transpose()
    w = np.linalg.inv(x_t.dot(x)).dot(x_t).dot(y)
    j = np.linalg.norm(np.subtract(x.dot(w), y))**2
    print('w vector:')
    print(w)
    print('J(w): '+ str(j))
    plt.scatter(x_1, y_1, color='red')
    line_x = np.linspace(min(x_1), max(x_1), 100)
    line_y = float(w[0][0])+float(w[1][0])*line_x
    plt.plot(line_x, line_y)
    plt.axis([min(x_1), max(x_1), min(y_1), max(y_1)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def gradient_descent(iterations, eta):
    print('Eta: '+str(eta))
    w = np.zeros(shape=(2,1)) #initialize w vector to 0 vector
    #w =np.array([0, 0])
    j_prev = 0
    conv = 0
    for i in range(iterations):
        conv +=1
        lst = []
        for k in range(len(w)): #each iteration 
            s = 0
            for z in range(len(x_1)): #calculate summation
                s=s+(((np.matmul(w.transpose(),x[z]))-y[z])*x[z][k])
            lst.append(w[k] - eta*s)
        w[0] = lst[0]
        w[1] = lst[1]
        j = np.linalg.norm(np.subtract(np.matmul(x, w), y))**2 #calculate norm 
        if abs(j_prev-j) < 0.0001: #see if j changed
            break
        j_prev = j
        
    print('w vector:')
    print(w)
    print('J(w): '+ str(j))
    print('Iterations till Convergence: '+str(conv))
    print()

def gradient_descent_part_d():
    print('Part D Eta: 0.05')
    print()
    eta = 0.05
    iterations = 40
    w = np.zeros(shape=(2,1)) #initialize w vector to 0 vector
    j_prev = 0
    
    for i in range(iterations):
        if i==0:
            line_x = np.linspace(min(x_1), max(x_1), 100)
            line_y = w[0][0]+w[1][0]*line_x
            print('Iteration: '+str(i if i==0 else i+1))
            print('w vector:')
            print(w)
            j = np.linalg.norm(np.subtract(x.dot(w), y))**2 #calculate norm
            print('J(w): '+ str(j))
            plt.scatter(x_1, y_1, color='red')
            plt.axis([min(x_1), max(x_1), min(y_1), max(y_1)])
            plt.plot(line_x, line_y)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        lst = []
        for k in range(len(w)): #each iteration 
            s = 0
            for z in range(len(x_1)): #calculate summation
                s=s+(((np.matmul(w.transpose(),x[z]))-y[z])*x[z][k])
            lst.append(w[k] - eta*s)
        w[0] = lst[0]
        w[1] = lst[1]
        j = np.linalg.norm(np.subtract(x.dot(w), y))**2 #calculate norm 
        if (i+1)%10==0:
            line_x = np.linspace(min(x_1), max(x_1), 100)
            line_y = w[0][0]+w[1][0]*line_x
            print('Iteration: '+str(i if i==0 else i+1))
            print('w vector:')
            print(w)
            print('J(w): '+ str(j))
            plt.scatter(x_1, y_1, color='red')
            plt.axis([min(x_1), max(x_1), min(y_1), max(y_1)])
            plt.plot(line_x, line_y)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
            print()
            
        if abs(j-j_prev) < 0.0001: #see if j changed
            break
        j_prev = j

def poly_regression(m):
    x_curr = x_1 
    y_curr = y 
    x_p = np.zeros(shape=(len(x_curr),m+1))
    for k in range(len(x_curr)):
        for i in range(m+1):
            x_p[k][i] = x_curr[k]**i
    x_t = x_p.transpose()
    global w_part_e
    w = np.linalg.inv(x_t.dot(x_p)).dot(x_t).dot(y_curr)
    w_part_e = w
    j = np.linalg.norm(np.subtract(x_p.dot(w), y_curr))**2
    return j

def test_poly_regression(m):
    x_curr = x_1_test 
    y_curr = y_test 
    x_p = np.zeros(shape=(len(x_curr),m+1))
    for k in range(len(x_curr)):
        for i in range(m+1):
            x_p[k][i] = x_curr[k]**i
    j = np.linalg.norm(np.subtract(x_p.dot(w_part_e), y_curr))**2
    return j

def generate_part_e_graph():
    n_train = len(x_1)
    n_test = len(x_1_test)
    rms_train = []
    rms_test = []
    for i in range(11):
        rms_train.append((poly_regression(i)/n_train)**0.5)
        rms_test.append((test_poly_regression(i)/n_test)**0.5)
    line_train, = plt.plot(rms_train, label='Training Data')
    line_test, = plt.plot(rms_test, label='Testing Data')
    plt.legend(handles=[line_train, line_test])
    plt.xlabel('m')
    plt.ylabel('Root Mean Square Error')
    
        
if __name__ == "__main__":

    load_data()


    #part d
    gradient_descent_part_d()

    #part e
    #generate_part_e_graph()