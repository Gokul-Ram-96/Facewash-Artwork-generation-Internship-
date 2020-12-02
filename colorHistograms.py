#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed Jul 22 13:02:59 2020

@author: aniketsadashiva , Gokul Ram
"""

import numpy as np

# Importing skimage tools for performing operations on image

from skimage.color import rgb2lab, deltaE_cie76  # deltaE_cie76 is used for computing Euclidean distance between two points in Lab color space

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import collections
from collections import namedtuple
from math import sqrt
import random
import os
try:
    import Image
except ImportError:
    from PIL import Image
import pandas as pd
dict_css4_colors = mcolors.CSS4_COLORS  # returns a dictionary of CSS4 colors.

def to_rgb(color):
    h =color.lstrip('#')
    return list(int(h[i:i+2], 16) for i in (0, 2, 4))

for item in dict_css4_colors:
    dict_css4_colors[item]=to_rgb(dict_css4_colors[item])




# Code For Performing K-Means Clustering to get Top 3 colors

Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))


def get_points(img):
    points = []
    (w, h) = img.size
    for (count, color) in img.getcolors(w * h):
        points.append(Point(color, 3, count))
    return points


rtoh = lambda rgb: '#%s' % ''.join('%02x' % p for p in rgb)


def find_top_colors(filename, n=3):
    png = Image.open(filename)


    png = Image.open(filename).convert('RGBA')
    bg = Image.new('RGBA', png.size, (255,255,255))

    img = Image.alpha_composite(bg, png)


    img.thumbnail((png.size))
    (w, h) = img.size

    points = get_points(img)
    clusters = kmeans(points, n, 1)
    rgbs = [map(int, c.center.coords) for c in clusters]
    return map(rtoh, rgbs)


def euclidean(p1, p2):
    return sqrt(sum([(p1.coords[i] - p2.coords[i]) ** 2 for i in
                range(p1.n)]))


def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 0
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += p.coords[i] * p.ct
    return Point([v / plen for v in vals], n, 1)


def kmeans(points, k, min_diff):
    clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]

    while 1:
        plists = [[] for i in range(k)]

        for p in points:
            smallest_distance = float('Inf')
            for i in range(k):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)

        diff = 0
        for i in range(k):
            old = clusters[i]
            center = calculate_center(plists[i], old.n)
            new = Cluster(plists[i], center, old.n)
            clusters[i] = new
            diff = max(diff, euclidean(old.center, new.center))

        if diff < min_diff:
            break

    return clusters

dicto={}
if __name__ == '__main__':
    directory = '/Users/aniketpihu/Desktop/ManageArtworks/t1'
    list_images = os.listdir(directory)
    list_colors = []
    for img in list_images:
        l2=[]

        img_source = directory + '/' + img
        if img == '.DS_Store':
            pass
        else:

            maps = find_top_colors(img_source)
            for i in list(maps):  # Changing map to list

                #print(i)
                list_colors.append(i)
                l2.append(i)
        dicto[img]=l2
    df1 = pd.DataFrame.from_dict(dicto,orient='index')
    df1.to_excel("ammonia.xlsx")
    print(df1)
    list_colors_rgb = []  # Storews


    for color in list_colors:
        list_colors_rgb.append(to_rgb(color))


    closest_fit_color = []  # Stores the names of colors closest to the given color using deltaE_cie76 metric for comparison
    closest_fit_color_name = ''
    dict_colors = dict_css4_colors.copy()

    for img_color in list_colors_rgb:
        dE = 1000
        img_color = np.uint8(np.asarray([[img_color]]))
        for std_colors in dict_colors:
            std_color = np.uint8(np.asarray([[dict_colors[std_colors]]]))
            diff = deltaE_cie76(rgb2lab(img_color), rgb2lab(std_color))
            if dE > diff:
                dE = diff
                closest_fit_color_name = std_colors

        closest_fit_color.append(closest_fit_color_name)


    counter_colors = collections.Counter(closest_fit_color)  # returns a dictionary corresponding to colors as key and frequency as values

    final_colors = []
    final_freq = []
    for item in counter_colors:
        final_colors.append(item)
        final_freq.append(counter_colors[item])


    fig = plt.figure()
    ax = fig.add_axes([0, 0, 10, 10])
    colors = []
    color_frequency = []
    for color in final_colors:
        colors.append(color)

    for color_freq in final_freq:
        color_frequency.append(color_freq)

    # Plotting the Histogram

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 10, 10])
    ax.bar(colors, color_frequency, color=colors, width=1)
    plt.xticks(rotation='vertical', fontsize=35)
    plt.yticks(np.arange(0, 60, step=5), fontsize=35)

    plt.show()
