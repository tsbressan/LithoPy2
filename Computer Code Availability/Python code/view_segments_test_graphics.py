import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
from skimage import filters
from skimage.util import img_as_float
import pandas as pd
import sys
import config

patch_out    = config.directory_output_images_U1480 #or config.directory_output_images_U1481


patch_out_1 = config.results_graphics + "dataset2/"

patch_RF = config.saveDir  + "/dataset2/" + sitee + "/"

dirr = []
for filename in os.listdir(patch_out):
    dirr.append (filename)
    
  
  
for PP in range(0,len(dirr)):
    try:
        os.mkdir(patch_out_1+dirr[PP])
        
    except OSError:
        print ("Creation of the directory %s failed" % dirr[PP])
    else:
        print ("Successfully created the directory %s " % dirr[PP])
        
                



for PP in range(0,len(dirr)):

    for filename in os.listdir(patch_out+dirr[PP]):
        nameFile = patch_out+dirr[PP]+"/"+filename
        
        filecsv = pd.read_csv(patch_RF+"xxxxx",sep=",") #xxxxx - adjust name file to test, example: join_RF_texture_test_51_1_50_90_0.4.csv
        filecsv = filecsv[filecsv["lithology"] == dirr[PP]]
        filecsv = filecsv[filecsv["name_image"] == filename]
        if (len(filecsv) > 0):
            region_       = filecsv["region"].values
            cod_lit_      = filecsv["cod_lit"].values
            cod_lit_test_ = filecsv["cod_lit_test"].values
            
            img = cv2.imread(nameFile)

            dpi=96
            height, width, depth = img.shape
            figsize = width / float(dpi), height / float(dpi)

            fig, ax = plt.subplots(ncols=1, nrows=1,figsize=figsize)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            segments  = slic(image, n_segments=400, compactness=10, sigma=1, multichannel=True,convert2lab=True,enforce_connectivity=True,slic_zero=0)
            segments  = segments  + 1

            
            for fi in range(0,len(region_)):
                region__    = region_[fi]
                
                for region in regionprops(segments):
                    la = region.label

                    if (int(region__) == int(la)):
                        
                        ce = region.centroid
                        

                        if (cod_lit_[fi] == cod_lit_test_[fi]):
                            rect = patches.Circle((ce[1], ce[0]), 20, color='green')

                        if (pd.isna(cod_lit_test_[fi])):
                            rect = patches.Circle((ce[1], ce[0]), 20, color='blue')

                        elif (cod_lit_[fi] != cod_lit_test_[fi]):
                            rect = patches.Circle((ce[1], ce[0]), 20, color='red')

                        ax.add_patch(rect)


            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = np.array(img)
            ax.imshow(pixels)        

            ax.imshow(mark_boundaries(image, segments))

            ax.axis("off")

            plt.tight_layout()
            #plt.show()
            fig.savefig(patch_out_1+dirr[PP]+"/"+filename,  bbox_inches='tight', edgecolor=None)
            print ("Processed image: ", filename)
            plt.close()
