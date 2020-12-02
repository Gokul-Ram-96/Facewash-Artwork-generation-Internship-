#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:37:24 2020

@author: Aniket, Gokul Ram , Komal Agnihotri
"""

import cv2
import pandas as pd
import numpy as np
import pdb
from PIL import Image

import os
if(os.path.exists('/Users/aniketpihu/Desktop/ManageArtworks/Nykaa_img/.DS_Store')):
    os.remove('/Users/aniketpihu/Desktop/ManageArtworks/Nykaa_img/.DS_Store')

def hex_to_rgb(color_hex):
    h = color_hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))


path_to_images='/Users/aniketpihu/Desktop/ManageArtworks/Nykaa_img'
list_img=os.listdir(path_to_images)
list_img_name=[]
for img in list_img:
    if img=='DS_Store':
        pass
    else:

        list_img_name.append((int(img.strip('.jpg'))-1))

#(path,sheet number)
df_img_analysis = pd.read_excel (r'/Users/aniketpihu/Desktop/ManageArtworks/artwork_analysis.xlsx',0)
df_text_coordinates= pd.read_excel (r'/Users/aniketpihu/Desktop/ManageArtworks/artwork_analysis.xlsx',1)
df_texts=pd.read_excel (r'/Users/aniketpihu/Desktop/ManageArtworks/artwork_analysis.xlsx',3)
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
df_text_coordinates=df_text_coordinates.drop('FIlename', 1)
df_texts=df_texts.drop('Filename', 1)


# set the rectangle background to white
rectangle_bgr = (255, 255, 255)
# make a black image
img = np.zeros((500, 500))
# set some text
text = "Some text in a box!"



#Draw rectangle for Artwork
xmin_artwork_coordinates=df_img_analysis['xmin_artwork']
xmax_artwork_coordinates=df_img_analysis['xmax_artwork']
ymin_artwork_coordinates=df_img_analysis['ymin_artwork']
ymax_artwork_coordinates=df_img_analysis['ymax_artwork']
xmin_image_coordinates=df_img_analysis['xmin_image']
xmax_image_coordinates=df_img_analysis['xmax_image']
ymin_image_coordinates=df_img_analysis['ymin_image']
ymax_image_coordinates=df_img_analysis['ymax_image']

colors=df_img_analysis['Color 3']
colors=[hex_to_rgb(color) for color in colors]
for filename in list_img_name:
    path_artwork='/Users/aniketpihu/Desktop/ManageArtworks/Nykaa/'+str(filename+1)+'.jpg'
    img_artwork = cv2.imread(path_artwork)


    crop_img = img_artwork[int(ymin_image_coordinates[filename]):int(ymax_image_coordinates[filename]), int(xmin_image_coordinates[filename]):int(xmax_image_coordinates[filename]),:]
    print(filename+1)


    artwork_sizes=[]

    artwork_sizes_w=[int(x_max)-int(x_min) for x_max,x_min in zip(xmax_artwork_coordinates,xmin_artwork_coordinates) ]
    artwork_sizes_h=[int(y_max)-int(y_min) for y_max,y_min in zip(ymax_artwork_coordinates,ymin_artwork_coordinates) ]



    # Reading an image in default mode
    img = np.zeros(shape=[artwork_sizes_h[filename], artwork_sizes_w[filename], 3], dtype=np.uint8)

    start_point=(0,0)
    end_point=(artwork_sizes_w[filename],artwork_sizes_h[filename])
    color=colors[filename]
    start_point_img=(int(xmin_image_coordinates[filename]),int(ymin_image_coordinates[filename]))
    end_point_img=(int(xmax_image_coordinates[filename]),int(ymax_image_coordinates[filename]))
    img2=cv2.rectangle(crop_img,start_point_img , end_point_img, color, -1)


    cv2.imwrite("cropped_img.png",img2)
    im1 = Image.open('/Users/aniketpihu/cropped_img.png')


    image00=cv2.rectangle(img,start_point , end_point, color, -1)
    cv2.imwrite("aw_rectangle.png",image00)

    image_pil = Image.open('/Users/aniketpihu/aw_rectangle.png')

    image_pil.paste(im1, (int(xmin_image_coordinates[filename])-int(xmin_artwork_coordinates[filename]), int(ymin_image_coordinates[filename])-int(ymin_artwork_coordinates[filename])))
    image_pil.save('/Users/aniketpihu/img_final.png', quality=100)

    image=cv2.imread('/Users/aniketpihu/img_final.png')

    #img[int(ymin_image_coordinates[filename]):int(ymax_image_coordinates[filename]), int(xmin_image_coordinates[filename]):int(xmax_image_coordinates[filename]),:]=img2

    text_and_coords=[[txt,coor] for txt,coor in zip(df_texts.loc[filename],df_text_coordinates.loc[filename]) if (txt is not (-1 or pd.isna((txt)))) and (coor != 0) ]

    #print(text_and_coords)
    for element in text_and_coords:
        text=element[0]
        coords=element[1].strip('][').split(', ')



        x=coords[0]
        y=coords[1]
        xmax=coords[2]
        ymax=coords[3]

        image = cv2.rectangle(image, (int(x), int(y)), (int(xmax), int(ymax)), (36,255,12), 1)
        cv2.putText(image, text, (int(x), int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        str_name=str(filename+1)+'.png'
        cv2.imwrite(str_name,image)

#cv2.imshow('test',image)
#cv2.imshow('test2',crop_img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
