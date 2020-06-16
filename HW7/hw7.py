#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 01:19:57 2020

@author: melodychen
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

MU_NUM = 4  # represents number of clusters
NUM_ITER = 10  # number of iterations we want K-means to run


def read_show_image():
    global img
    img = cv2.imread('./UCLA_Bruin.jpg')
    img = img.astype('float')
    # Part A: uncomment below to print original image
    # plt.imshow(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB))
    # plt.show()


def k_means_algorithm():
    K = []
    points = []
    # find mu_1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            points.append(img[i][j])
            if img[i][j][0] == 250 and img[i][j][1] == 249 and img[i][j][2] == 229:
                K.append([i, j])
    # find rest of mus
    for z in range(MU_NUM-1):
        mu = [0,0]
        dist = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dists = []
                for n in range(z+1):
                    dists.append(np.linalg.norm(img[i][j] - img[K[n][0]][K[n][1]]))
                if min(dists) > dist:
                    dist = min(dists)
                    mu = [i, j]
        K.append(mu)
    # run actual K means for 10 iterations
    mus = [img[K[n][0]][K[n][1]] for n in range(MU_NUM)]  # stores BGR value of cluster centers
    Js = []  # stores J value of objective function for each iteration
    for iter in range(NUM_ITER):
        r = []  # stores which cluster each point belongs to
        for point in points:
            dist = 3*(255**2)  # max value distance can be
            cluster = -1
            # find cluster that is closest
            for n in range(MU_NUM):
                curr_dist = np.linalg.norm(point - mus[n])
                if curr_dist < dist:
                    dist = curr_dist
                    cluster = n
            r.append(cluster)
        # find the new center
        for n in range(MU_NUM):
            summation = np.zeros(3)  # represents summation of x vectors in formula for finding new center
            count = 0  # represents number of times point belong to this particular cluster
            for index, point in enumerate(points):
                if r[index] == n:
                    summation += point
                    count += 1
            mus[n] = summation/count
        J = 0
        # calculate J's, find norm of every point distance from cluster
        for index, point in enumerate(points):
            J += np.linalg.norm(point - mus[r[index]])**2
        Js.append(J)
        print("Iteration "+str(iter)+" done!")
    print("Cluster Centers (B,G,R): ")
    print(mus)
    print("J after each iteration: ")
    print(Js)

    # Part B: plot Js vs. Iterations, comment code below to see Part C plot

    line_test, = plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Js, label='J')
    plt.legend(handles=[line_test])
    plt.title("Plot of J for K=" + str(MU_NUM))
    plt.xlabel('# Iterations')
    plt.ylabel('J')
    plt.show()

    # Part C: replace cluster with center pixel, uncomment code below to see Part C plot and comment code above

    # for n in range(120000):
    #     points[n] = mus[r[n]]
    # new_img = np.array(points)
    # new_img = np.reshape(new_img, img.shape)
    # plt.title("Plot of Compressed Image for K=" + str(MU_NUM))
    # plt.imshow(cv2.cvtColor(new_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    # plt.show()


if __name__ == "__main__":
    read_show_image()  # Part A
    k_means_algorithm()  # Part B and C

