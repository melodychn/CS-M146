#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 07:52:39 2020

@author: melodychen
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from sklearn import tree
from sklearn.metrics import accuracy_score

def count_majority(file_name):
    print(file_name+": ")
    count_1 = 0
    count_0 = 0
    with open('data'+str(file_name)+'_Y.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(float(row[0])==1):
                count_1+=1
            else:
                count_0+=1
    print("1 Count: "+str(count_1))
    print("0 Count: "+str(count_0))
    if(count_1 > count_0):
        print("Accuracy: "+str(count_1/(count_0+count_1)))
    else:
        print("Accuracy: "+str(count_0/(count_0+count_1)))

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
            x.append([row[0], row[1], row[2], row[3], row[4], row[5]])
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

def build_decision_tree():
    tree_builder = tree.DecisionTreeClassifier(criterion='entropy')
    tree_builder = tree_builder.fit(x, y)
    tr_acc = tree_builder.score(x, y)
    test_acc = tree_builder.score(x_test,y_test)
    print("Training Accuracy: "+str(tr_acc))
    print("Testing Accuracy: "+str(test_acc))
if __name__ == "__main__":
    count_majority("Training")
    count_majority("Testing")
    load_data()
    build_decision_tree()
    
        


          
        