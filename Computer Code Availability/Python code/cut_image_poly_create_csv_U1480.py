import json
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_float
import pandas as pd
import sys


import config

#variables of file config.py
path      = config.directory_IODP_images_U1480
patch_out = config.directory_output_images_U1480 


patch_out_description = config.directory_geo_prop_orig_interp_U1480
patch_properties      = config.directory_geo_prop_orig_U1480
litho = config.litho


prop = ['gra','mad','ms','ngr','pwl','rgb','rsc']

hole = ['e','f','g','h'] #run one by one


#save the contents of the complete arrays
np.set_printoptions(threshold=sys.maxsize)
#-------------------------------------


#read files in directory
data = []
files = []
filesP = []
size_files = []
for filename in os.listdir(path):
    files.append (filename)
    size_files.append (os.path.getsize(path+"/"+filename))

    
for filenameP in os.listdir(patch_properties):
    filesP.append (filenameP)
#========================    
    
#create directory of output per lithology
for PP in range(0,len(litho)):
    try:
        os.mkdir(patch_out+litho[PP])
        
    except OSError:
        print ("Creation of the directory %s failed" % litho[PP])
    else:
        print ("Successfully created the directory %s " % litho[PP])
#========================================
        
        
pts = []
#OPEN JSON FILE WITH IMAGE POLYGONS
with open(config.file_json_U1480) as json_file:
    data = json.load(json_file)
    for p in range(len(data)):
        col = str(files[p]) + str(size_files[p])
        split_col = col.split('-')
        hole_ = split_col[1]
        
        #if (hole_[5:6] == hole[0]):
            
        if (data[col]['regions']):
        
            for rr in range(0,len(data[col]['regions'])):
                
                
                X = data[col]['regions'][rr]['shape_attributes']['all_points_x']
                Y = data[col]['regions'][rr]['shape_attributes']['all_points_y']
                Lithology = data[col]['regions'][rr]['region_attributes']['Lithology']
                
                
                
                primeiroFT = files[p].split('_')
                name_test_exist = str(primeiroFT[0])+";"+"region-"+str(rr)+";"+str(Lithology)+".csv"
                test = 0
                for PP in range(0,len(filesP)):
                    if (name_test_exist == filesP[PP]):
                        test = 1
                
                if (test == 0):
                    for i in range(0,len(X)):
                        pts.append(([X[i],Y[i]]))


                    img = cv2.imread(path+"/"+files[p])
                    
                    
                    
                    pts1 = np.array(pts)
                    
                    
                    mask = np.zeros((img.shape[0], img.shape[1]))
                    cv2.fillConvexPoly(mask, pts1, 1)
                    mask = mask.astype(np.bool)
                    out = np.zeros_like(img)
                    out[mask] = img[mask]
                    
                    #saves the image with the polygon already defined
                    for folders_litho in os.listdir(patch_out):
                        
                        if (folders_litho == Lithology):
                            
                            primeiro = files[p].split('_')
                            name_img_new = str(primeiro[0])+";"+'region-'+str(rr)+';'+str(Lithology)+'.png'
                            name_file_f = patch_out + folders_litho + "/" + str(primeiro[0])+";"+"region-"+str(rr)+";"+str(Lithology)+".png" 
                            
                            cv2.imwrite(name_file_f, out)
                            #print (out)
                            print (name_file_f)


                    pts = [] 
                    
                    #####################################
                    cols2 = ['Lithology', 'Prop', 'value', 'Exp','SiteHole','CoreType','Sect','Offset']
                    data_csv_train = []

                    Depth = data[col]['regions'][rr]['region_attributes']['Depth']
                    
                    primeiro1 = Depth.split('_')

                    OffsetI = primeiro1[0].replace(',','.')
                    OffsetI = float(OffsetI)

                    OffsetF = primeiro1[1].replace(',','.')
                    OffsetF = float(OffsetF)


                    #separate the file name
                    split_file = files[p].split('-')
                    ExpF      = split_file[0]
                    SiteHoleF = split_file[1]
                    CoreTypeF = split_file[2]
                    SectF     = split_file[3]

                    

                    for f in range(0,len(prop)):
                        
                        fileP = pd.read_csv(patch_properties+prop[f]+".csv",sep=",")

                        for fi in range(0,len(fileP)):
                            Exp    = str(fileP['Exp'][fi]) #ok

                            Site   = str(fileP['Site'][fi]) # Site + Hole
                            Site = Site.lower()
                            Hole   = str(fileP['Hole'][fi])
                            Hole = Hole.lower()
                            SiteHole = Site + Hole

                            Core   = str(fileP['Core'][fi]) #Core + Type
                            Type   = str(fileP['Type'][fi])
                            Type = Type.lower()

                            CoreType = Core + Type

                            Sect   = str(fileP['Sect'][fi])
                            Offset = fileP['Offset'][fi]

                            if Exp == ExpF and SiteHole == SiteHoleF and CoreType == CoreTypeF and Sect == SectF and Offset >= OffsetI and Offset <= OffsetF:
                                print (SiteHole+"-"+CoreType+"-"+Sect+"---"+str(Offset) + "--------" + "Interval:"+str(OffsetI)+"-"+str(OffsetF))
                                if prop[f] == 'gra':
                                    data_csv_train.append ([Lithology, prop[f], fileP['Bulkdensity'][fi], Exp, SiteHole, CoreType, Sect, Offset])
                                elif prop[f] == 'mad': 
                                    #concatenar valores
                                    value_1 = str(fileP['Moisturedry(wt%)'][fi])+"_"+str(fileP['Moisturewet(wt%)'][fi])+"_"+str(fileP['Bulkdensity(g/cm)'][fi])+"_"+str(fileP['Drydensity(g/cm)'][fi])+"_"+str(fileP['Graindensity(g/cm)'][fi])+"_"+str(fileP['Porosity(vol%)'][fi])+"_"+str(fileP['Voidratio'][fi])
                                    data_csv_train.append ([Lithology, prop[f], value_1, Exp, SiteHole, CoreType, Sect, Offset])
                                elif prop[f] == 'ms': 
                                    data_csv_train.append ([Lithology, prop[f], fileP['Magneticsusceptibility'][fi], Exp, SiteHole, CoreType, Sect, Offset])                            
                                elif prop[f] == 'ngr':  
                                    data_csv_train.append ([Lithology, prop[f], fileP['NGRtotalcounts'][fi], Exp, SiteHole, CoreType, Sect, Offset])
                                elif prop[f] == 'pwl':  
                                    data_csv_train.append ([Lithology, prop[f], fileP['Pwavevelocity'][fi], Exp, SiteHole, CoreType, Sect, Offset])
                                elif prop[f] == 'rgb':
                                    value_2 = str(fileP['R'][fi])+"_"+str(fileP['G'][fi])+"_"+str(fileP['B'][fi])
                                    data_csv_train.append ([Lithology, prop[f], value_2, Exp, SiteHole, CoreType, Sect, Offset])
                                elif prop[f] == 'rsc':  
                                    value_3 = str(fileP['ReflectanceL'][fi])+"_"+str(fileP['Reflectancea'][fi])+"_"+str(fileP['Reflectanceb'][fi])
                                    data_csv_train.append ([Lithology, prop[f], value_3, Exp, SiteHole, CoreType, Sect, Offset])

                    dt_result_csv = pd.DataFrame(data_csv_train,columns=cols2)
                    primeiroF = files[p].split('_')
                    dt_result_csv.to_csv(patch_out_description + "dataset1/" + str(primeiroF[0])+";"+"region-"+str(rr)+";"+str(Lithology)+".csv")
                    #---------------------------------
                        


