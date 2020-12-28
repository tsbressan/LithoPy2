import numpy as np
import json
import os
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
from skimage.measure import shannon_entropy
from skimage import filters
import cv2
from skimage.segmentation import slic
from skimage.measure import regionprops
import config

path                  = config.directory_IODP_images_U1480
patch_out             = config.directory_output_images_U1480

patch_out_description = config.directory_geo_prop_orig_interp_U1480


datasets_RF = config.datasets_RF
dataset1_U1480_group1 = datasets_RF+"dataset1/U1480/group1/"


data = []
data1 = []
new_data = []



cols = ['name_image','region',
'C_D_1_an_0','C_D_1_an_45','C_D_1_an_90','C_D_1_an_135',
'C_D_2_an_0','C_D_2_an_45','C_D_2_an_90','C_D_2_an_135',
'C_D_3_an_0','C_D_3_an_45','C_D_3_an_90','C_D_3_an_135',
'Di_D_1_an_0','Di_D_1_an_45','Di_D_1_an_90','Di_D_1_an_135',
'Di_D_2_an_0','Di_D_2_an_45','Di_D_2_an_90','Di_D_2_an_135',
'Di_D_3_an_0','Di_D_3_an_45','Di_D_3_an_90','Di_D_3_an_135',
'H_D_1_an_0','H_D_1_an_45','H_D_1_an_90','H_D_1_an_135',
'H_D_2_an_0','H_D_2_an_45','H_D_2_an_90','H_D_2_an_135',
'H_D_3_an_0','H_D_3_an_45','H_D_3_an_90','H_D_3_an_135',
'A_D_1_an_0','A_D_1_an_45','A_D_1_an_90','A_D_1_an_135',
'A_D_2_an_0','A_D_2_an_45','A_D_2_an_90','A_D_2_an_135',
'A_D_3_an_0','A_D_3_an_45','A_D_3_an_90','A_D_3_an_135',
'E_D_1_an_0','E_D_1_an_45','E_D_1_an_90','E_D_1_an_135',
'E_D_2_an_0','E_D_2_an_45','E_D_2_an_90','E_D_2_an_135',
'E_D_3_an_0','E_D_3_an_45','E_D_3_an_90','E_D_3_an_135',
'Cor_D_1_an_0','Cor_D_1_an_45','Cor_D_1_an_90','Cor_D_1_an_135',
'Cor_D_2_an_0','Cor_D_2_an_45','Cor_D_2_an_90','Cor_D_2_an_135',
'Cor_D_3_an_0','Cor_D_3_an_45','Cor_D_3_an_90','Cor_D_3_an_135','mean_intensity','lithology','cod_lit']

cols1 = ['name_image','label','mean_offset','value_1_gra','value_1_mad','value_1_ms','value_1_ngr','value_1_pwl','value_1_rgb','value_1_rsc','value_2_mad','value_2_rgb','value_2_rsc','value_3_mad','value_3_rgb','value_3_rsc','value_4_mad','value_5_mad','value_6_mad']


files = []
size_files = []
for filename in os.listdir(path):
    files.append (filename)
    size_files.append (os.path.getsize(path+"/"+filename))


    
files_output_litho=[]    
for filename1 in os.listdir(patch_out):
    files_output_litho.append (filename1)
    
for folders_litho in os.listdir(patch_out):
    if (folders_litho == 'Ash_tuff'):  #to perform all lithologies, remove or adjust if ()

        
        for fcod in range (0,len(files_output_litho)):
            if (files_output_litho[fcod] == folders_litho):
                cod_lit = fcod
                
        for files_img_litho in os.listdir(patch_out+"/"+folders_litho):

            image = patch_out+"/"+folders_litho+"/"+files_img_litho
            
            name_ = str(files_img_litho)
            print (name_)
            
            #read image
            img1 = cv2.imread(image)
            

            #split 
            split_img = files_img_litho.split(';')
            if (len(split_img[1]) > 8):
                region_ = split_img[1][7:9]
            else:
                region_ = split_img[1][7]
                
            litho_ = split_img[2]
            litho__ = litho_.split('.')

            data_core = split_img[0].split('-')
            exp      = data_core[0]
            sitehole = data_core[1]
            coreType = data_core[2]
            sect     = data_core[3]
            
            segments  = slic(img1, n_segments=400, compactness=10, sigma=1, multichannel=True,convert2lab=True,enforce_connectivity=True)
            segments  = segments  + 1  
      

            mean_0 = []
            mean_1 = []
            mean_2 = []
            mean_full = []
            areaT = 0
            info_bbox_centroid = []
            info_depth_img = []
            regions0 = regionprops(segments, intensity_image=img1[:,:,0])
            regions1 = regionprops(segments, intensity_image=img1[:,:,1])
            regions2 = regionprops(segments, intensity_image=img1[:,:,2])
            for r in regions0:
                mean_0.append(r.mean_intensity)
            for rr in regions1:
                mean_1.append(r.mean_intensity)
            for rrr in regions2:
                mean_2.append(r.mean_intensity)

            count = len(np.unique(segments))
            for c in range (0,count):
                mean_full.append((mean_0[c]+mean_1[c]+mean_2[c])/3)
                
            for cc in range (0,len(mean_full)):
                if (mean_full[cc] > 1):


                    for region in regionprops(segments):
                        if (region.label == cc+1):
                            label1 = int(region.label)
                            
                            seg_ = img_as_ubyte(region.convex_image)
                            grayImg = img_as_ubyte(color.rgb2gray(seg_))

                            dist = [1,2,3]
                            an = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                            prop = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']


                            glcm = greycomatrix(grayImg, 
                                                distances=dist, 
                                                angles=an,
                                                symmetric=False,
                                                normed=False)
                               
                               
                            data.append(name_)
                            data.append(label1)
                            
                            for p in prop:
                            
                                #print (p)
                                result_ = (greycoprops(glcm, p))
                                for i in range (0,len(dist)):
                                    for x in range (0,len(an)):
                                        data.append(result_[i][x])
                                #print ("---")
                                    
                            data.append(mean_full[cc])
                            data.append(folders_litho)
                            data.append(cod_lit)


                            data1.append(data)
                            data=[]
                            
                            #print ("Region:", region.label, "dissimilarity:", dis)
                            #print ("bouding box: ",region.bbox)
                            #print ("Centroid: ",region.centroid)
                            
                            areaT = areaT+region.area
                            #print ("-------")
                            #save properties regions
                            area_cm = ((region.area*0.02646)/1000)
                            area_cm = round(area_cm, 2)
                            info_bbox_centroid.append([files_img_litho,region.label,region.bbox,region.centroid,region.area,area_cm])
    
    
    
            #json Total Area in cm _ min x max
            with open(config.file_json_U1480) as json_file:
                data2 = json.load(json_file)
                for p in range(len(data2)):
                    col = str(files[p]) + str(size_files[p])
                    split_col = data2[col]['filename'].split('_')
                    if (split_col[0] == split_img[0]):
                        split_col = col.split('-')
                        hole_ = split_col[1]
                        

                        if (data2[col]['regions']):
                            
                            for rr in range(0,len(data2[col]['regions'])):

                                if (rr == int(region_)):
                                    
                                    Lithology = data2[col]['regions'][rr]['region_attributes']['Lithology']
                                    Litho__ = Lithology+".png"
                                    
                                    if (Litho__ == litho_):
                                        #print (Lithology)
                                        X = data2[col]['regions'][rr]['shape_attributes']['all_points_x']
                                        Y = data2[col]['regions'][rr]['shape_attributes']['all_points_y']
                                        Depth = data2[col]['regions'][rr]['region_attributes']['Depth']
                                        info_depth_img.append ([files_img_litho,X,Y,Depth])
    
    
    
    
            #mix_max collumn and length area in cm

            X = info_depth_img[0][1]
            Y = info_depth_img[0][2]
            Depth = info_depth_img[0][3]
            max_Y = float(max(Y))
            min_Y = float(min(Y))
            split_Dp = Depth.split('_')
            min_Dp = float(split_Dp[0])
            max_Dp = float(split_Dp[1])
            Depth_ar = float(max_Dp) - float(min_Dp)
            value_per_line = Depth_ar/(max_Y - min_Y)
            
            info_full_cm_area = []
            for i in range (0,len(info_bbox_centroid)):  
                
                bbox_rg = info_bbox_centroid[i][2]
                max_Y_bbox = float(bbox_rg[2])
                min_Y_bbox = float(bbox_rg[0])
                max_min_bbox = max_Y_bbox - min_Y_bbox
                
                #calcules
                start_leng = float(((min_Y_bbox - min_Y)*value_per_line)+min_Dp)       
                end_leng = float(((max_Y_bbox-min_Y_bbox)*value_per_line)+start_leng)       
                start_leng = round(start_leng, 2)
                end_leng = round(end_leng, 2)
                #--------
                info_full_cm_area.append([info_bbox_centroid[i][0],info_bbox_centroid[i][1],start_leng,end_leng,info_bbox_centroid[i][5]])
    
    
    
            
            for ii in range (0,len(info_full_cm_area)):
                nome  = info_full_cm_area[ii][0]
                label = info_full_cm_area[ii][1]
                cm_start = float(info_full_cm_area[ii][2])
                cm_end = float(info_full_cm_area[ii][3])
                
                sum_rsc2   = 0#
                sum_mad2   = 0#
                sum_rsc3   = 0#
                
                sum_mad1   = 0
                sum_ms     = 0
                sum_ngr    = 0
                sum_pwl    = 0
                sum_gra    = 0
                sum_rgb1   = 0
                sum_rgb2   = 0
                sum_rgb3   = 0
                sum_rsc1   = 0
                sum_mad3   = 0
                sum_mad4   = 0
                sum_mad5   = 0
                sum_mad6   = 0
                
                sum_offset = 0
                qtd = 0
                
                mean_rsc2  = 0
                mean_rsc3  = 0
                mean_mad2  = 0
                
                mean_mad1  = 0
                mean_ms    = 0
                mean_ngr   = 0
                mean_pwl   = 0
                mean_gra   = 0
                mean_rgb1  = 0
                mean_rgb2  = 0
                mean_rgb3  = 0
                mean_rsc1  = 0
                mean_mad3  = 0
                mean_mad4  = 0
                mean_mad5  = 0
                mean_mad6  = 0
                
                mean_offset= 0
                
                filecsv = pd.read_csv(dataset1_U1480_group1+"dataset1_U1480_group1.csv",sep=",")
                for fi in range(0,len(filecsv)):
                    
                    Exp_    = str(filecsv['Exp'][fi]) #ok

                    Site   = str(filecsv['Site'][fi]) # Site + Hole
                    Hole   = str(filecsv['Hole'][fi])
                    SiteHole_ = Site + Hole

                    Core   = str(filecsv['Core'][fi]) #Core + Type
                    Type   = str(filecsv['Type'][fi])
                    CoreType_ = Core + Type

                    Sect_   = str(filecsv['Sect'][fi])
                    lithology_   = str(filecsv['lithology'][fi])
                                  
                    if (Exp_ == exp) and (SiteHole_ == sitehole) and (CoreType_ == coreType) and (Sect_ == sect) and (lithology_ == litho__[0]):
                        
                        Offset = float(filecsv['Offset'][fi])
                        if (Offset >= cm_start) and (Offset <= cm_end):
                            
                            value_2_rsc = float(filecsv['value_2_rsc'][fi])
                            value_1_mad = float(filecsv['value_1_mad'][fi])
                            value_1_ms  = float(filecsv['value_1_ms'][fi])
                            value_1_ngr = float(filecsv['value_1_ngr'][fi])
                            value_1_pwl = float(filecsv['value_1_pwl'][fi])
                            value_2_mad = float(filecsv['value_2_mad'][fi])
                            value_3_rsc = float(filecsv['value_3_rsc'][fi])
                            value_1_gra = float(filecsv['value_1_gra'][fi])
                            value_1_rgb = float(filecsv['value_1_rgb'][fi])
                            value_2_rgb = float(filecsv['value_2_rgb'][fi])
                            value_3_rgb = float(filecsv['value_3_rgb'][fi])
                            value_1_rsc = float(filecsv['value_1_rsc'][fi])
                            value_3_mad = float(filecsv['value_3_mad'][fi])
                            value_4_mad = float(filecsv['value_4_mad'][fi])
                            value_5_mad = float(filecsv['value_5_mad'][fi])
                            value_6_mad = float(filecsv['value_6_mad'][fi])
                            
                            qtd = qtd + 1
                            sum_rsc2   = sum_rsc2 + value_2_rsc
                            
                            sum_mad1   = sum_mad1 + value_1_mad
                            sum_ms     = sum_ms + value_1_ms
                            sum_ngr    = sum_ngr + value_1_ngr
                            sum_pwl    = sum_pwl + value_1_pwl
                            
                            sum_rsc3   = sum_rsc3 + value_3_rsc
                            sum_mad2   = sum_mad2 + value_2_mad
                            
                            sum_gra    = sum_gra + value_1_gra
                            sum_rgb1   = sum_rgb1 + value_1_rgb
                            sum_rgb2   = sum_rgb2 + value_2_rgb
                            sum_rgb3   = sum_rgb3 + value_3_rgb
                            sum_rsc1   = sum_rsc1 + value_1_rsc
                            sum_mad3   = sum_mad3 + value_3_mad
                            sum_mad4   = sum_mad4 + value_4_mad
                            sum_mad5   = sum_mad5 + value_5_mad
                            sum_mad6   = sum_mad6 + value_6_mad
                            
                            sum_offset = sum_offset + Offset
                            
                            
                if (qtd > 0):            
                    mean_rsc2   = float(sum_rsc2)/float(qtd) 
                    mean_rsc3   = float(sum_rsc3)/float(qtd)  
                    mean_mad2    = float(sum_mad2)/float(qtd)  
                    
                    mean_mad1    = float(sum_mad1)/float(qtd) 
                    mean_ms    = float(sum_ms)/float(qtd) 
                    mean_ngr    = float(sum_ngr)/float(qtd) 
                    mean_pwl    = float(sum_pwl)/float(qtd) 
                    mean_gra    = float(sum_gra)/float(qtd) 
                    mean_rgb1    = float(sum_rgb1)/float(qtd) 
                    mean_rgb2    = float(sum_rgb2)/float(qtd) 
                    mean_rgb3    = float(sum_rgb3)/float(qtd) 
                    mean_rsc1    = float(sum_rsc1)/float(qtd) 
                    mean_mad3    = float(sum_mad3)/float(qtd) 
                    mean_mad4    = float(sum_mad4)/float(qtd) 
                    mean_mad5    = float(sum_mad5)/float(qtd) 
                    mean_mad6    = float(sum_mad6)/float(qtd) 
                    
                    mean_offset = float(sum_offset)/float(qtd)  
                    
                    mean_rsc2   = round(mean_rsc2, 2)
                    mean_rsc3   = round(mean_rsc3, 2)
                    mean_mad2    = round(mean_mad2, 2)
                    
                    mean_mad1    = round(mean_mad1, 2)
                    mean_ms      = round(mean_ms, 2)
                    mean_ngr     = round(mean_ngr, 2)
                    mean_pwl     = round(mean_pwl, 2)
                    mean_gra     = round(mean_gra, 2)
                    mean_rgb1    = round(mean_rgb1, 2)
                    mean_rgb2    = round(mean_rgb2, 2)
                    mean_rgb3    = round(mean_rgb3, 2)
                    mean_rsc1    = round(mean_rsc1, 2)
                    mean_mad3    = round(mean_mad3, 2)
                    mean_mad4    = round(mean_mad4, 2)
                    mean_mad5    = round(mean_mad5, 2)
                    mean_mad6    = round(mean_mad6, 2)
                    
                    mean_offset = round(mean_offset, 2)
                    

                new_data.append([nome,label,mean_offset,mean_gra,mean_mad1,mean_ms,mean_ngr,mean_pwl,mean_rgb1,mean_rsc1,mean_mad2,mean_rgb2,mean_rsc2,mean_mad3,mean_rgb3,mean_rsc3,mean_mad4,mean_mad5,mean_mad6])
                
                qtd=0
    
    

        dt_result = pd.DataFrame(data1,columns=cols)
        dt_result1 = pd.DataFrame(new_data,columns=cols1)
        

        nameFileExp1 = patch_out_description +"dataset2/extract_textures_region_"+folders_litho+"_v01_fullProper.csv"
        nameFileExp2 = patch_out_description +"dataset2/extract_textures_region_"+folders_litho+"_v02_fullProper.csv"
        nameFileExpF = patch_out_description +"dataset2/extract_textures_region_"+folders_litho+"_v_join_fullProper.csv"
            
            
        dt_result.to_csv(nameFileExp1)
        dt_result1.to_csv(nameFileExp2)
        
        filecsv1 = pd.read_csv(nameFileExp1,sep=",")
        filecsv2 = pd.read_csv(nameFileExp2,sep=",")
        
        result = pd.concat([filecsv1, filecsv2], axis=1, join='inner')
        result.to_csv(nameFileExpF)
        print ("File create:", folders_litho)
        data1 = []

