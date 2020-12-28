
import pandas as pd
import numpy as np
import shutil
import config
import warnings

warnings.filterwarnings("ignore")


#to dataset0

patch_properties = config.directory_geo_prop_orig_interp_U1480 + "dataset1/"
datasets_RF = config.datasets_RF + "dataset0/"

method_inter = ["linear","polynomial_order_1","polynomial_order_2","polynomial_order_3","from_derivatives","spline_order_1","spline_order_2","spline_order_3","spline_order_4","slinear","quadratic","cubic","akima","pchip","piecewise_polynomial"]   
exp = ['362']
site = ['U1480']
hole = ['E']

for u in range (0,len(method_inter)):
    for x in range(0,len(exp)):
        for xx in range(0,len(site)):
            for xxx in range(0,len(hole)):
            
                name_file = method_inter[u]+"_full_"+str(exp[x])+"_"+str(site[xx])+"_"+str(hole[xxx])+".csv"

                
                shutil.copyfile(patch_properties+name_file, datasets_RF+name_file)
                
                
                
print ("Create dataset0")
#to dataset1-U1480

datasets_RF = config.datasets_RF + "dataset1/U1480/group1/"

method_inter = ["akima"]   
exp = ['362']
site = ['U1480']
hole = ['E','F','G','H']


                
name_file_E = method_inter[0]+"_full_"+str(exp[0])+"_"+str(site[0])+"_"+str(hole[0])+".csv"
name_file_F = method_inter[0]+"_full_"+str(exp[0])+"_"+str(site[0])+"_"+str(hole[1])+".csv"
name_file_G = method_inter[0]+"_full_"+str(exp[0])+"_"+str(site[0])+"_"+str(hole[2])+".csv"
name_file_H = method_inter[0]+"_full_"+str(exp[0])+"_"+str(site[0])+"_"+str(hole[3])+".csv"

filePPP_E = pd.read_csv(patch_properties+name_file_E,sep=",")
filePPP_F = pd.read_csv(patch_properties+name_file_F,sep=",")
filePPP_G = pd.read_csv(patch_properties+name_file_G,sep=",")
filePPP_H = pd.read_csv(patch_properties+name_file_H,sep=",")

join_fileP = [filePPP_E , filePPP_F , filePPP_G , filePPP_H]

name_file = datasets_RF+"dataset1_U1480_group1.csv"
result = pd.concat(join_fileP)
result.to_csv(name_file)
print ("Create dataset1-U1480-Group1")

result = pd.read_csv(name_file,sep=",")

#group 2
for x in range(0,len(result)):
    
    if (result['cod_lit'][x] == 0) or (result['cod_lit'][x] == 1) or (result['cod_lit'][x] == 10) :
        result['cod_lit'][x] = 0
        
    if (result['cod_lit'][x] == 2):
        result['cod_lit'][x] = 1
        
    if (result['cod_lit'][x] == 3) or (result['cod_lit'][x] == 4):
        result['cod_lit'][x] = 2

    if (result['cod_lit'][x] == 5):
        result['cod_lit'][x] = 3
        
    if (result['cod_lit'][x] == 11) or (result['cod_lit'][x] == 12):
        result['cod_lit'][x] = 4
        
    if (result['cod_lit'][x] == 6) or (result['cod_lit'][x] == 8) or (result['cod_lit'][x] == 9) or (result['cod_lit'][x] == 13):
        result['cod_lit'][x] = 5

datasets_RF = config.datasets_RF + "dataset1/U1480/group2/"
name_file = datasets_RF+"dataset1_U1480_group2.csv"   
result.to_csv(name_file)
print ("Create dataset1-U1480-Group2")        

        
#to dataset1-U1481
patch_properties = config.directory_geo_prop_orig_interp_U1481 + "dataset1/"       
        
datasets_RF = config.datasets_RF + "dataset1/U1481/group1/"

method_inter = ["akima"]   
exp = ['362']
site = ['U1481']
hole = ['A']


                
name_file_A = method_inter[0]+"_full_"+str(exp[0])+"_"+str(site[0])+"_"+str(hole[0])+".csv"


filePPP_A = pd.read_csv(patch_properties+name_file_A,sep=",")

join_fileA = [filePPP_A]

name_file = datasets_RF+"dataset1_U1481_group1.csv"
result = pd.concat(join_fileA)
result.to_csv(name_file)      
print ("Create dataset1-U1481-Group1") 

result = pd.read_csv(name_file,sep=",")

#group 2
for x in range(0,len(result)):
    
    if (result['cod_lit'][x] == 0) or (result['cod_lit'][x] == 1) or (result['cod_lit'][x] == 10) :
        result['cod_lit'][x] = 0
        
    if (result['cod_lit'][x] == 2):
        result['cod_lit'][x] = 1
        
    if (result['cod_lit'][x] == 3) or (result['cod_lit'][x] == 4):
        result['cod_lit'][x] = 2

    if (result['cod_lit'][x] == 5):
        result['cod_lit'][x] = 3
        
    if (result['cod_lit'][x] == 11) or (result['cod_lit'][x] == 12):
        result['cod_lit'][x] = 4
        
    if (result['cod_lit'][x] == 6) or (result['cod_lit'][x] == 8) or (result['cod_lit'][x] == 9) or (result['cod_lit'][x] == 13):
        result['cod_lit'][x] = 5

datasets_RF = config.datasets_RF + "dataset1/U1481/group2/"
name_file = datasets_RF+"dataset1_U1481_group2.csv"   
result.to_csv(name_file)          

print ("Create dataset1-U1481-Group2") 






#to dataset2-U1480
patch_out_description = config.directory_geo_prop_orig_interp_U1480

litho = config.litho

nameFileExp0 = patch_out_description +"dataset2/extract_textures_region_"+litho[0]+"_v_join_fullProper.csv"
nameFileExp1 = patch_out_description +"dataset2/extract_textures_region_"+litho[1]+"_v_join_fullProper.csv"
nameFileExp2 = patch_out_description +"dataset2/extract_textures_region_"+litho[2]+"_v_join_fullProper.csv"
nameFileExp3 = patch_out_description +"dataset2/extract_textures_region_"+litho[3]+"_v_join_fullProper.csv"

nameFileExp5 = patch_out_description +"dataset2/extract_textures_region_"+litho[5]+"_v_join_fullProper.csv"
nameFileExp6 = patch_out_description +"dataset2/extract_textures_region_"+litho[6]+"_v_join_fullProper.csv"

nameFileExp8 = patch_out_description +"dataset2/extract_textures_region_"+litho[8]+"_v_join_fullProper.csv"
nameFileExp9 = patch_out_description +"dataset2/extract_textures_region_"+litho[9]+"_v_join_fullProper.csv"
nameFileExp10 = patch_out_description +"dataset2/extract_textures_region_"+litho[10]+"_v_join_fullProper.csv"
nameFileExp11 = patch_out_description +"dataset2/extract_textures_region_"+litho[11]+"_v_join_fullProper.csv"
nameFileExp12 = patch_out_description +"dataset2/extract_textures_region_"+litho[12]+"_v_join_fullProper.csv"
nameFileExp13 = patch_out_description +"dataset2/extract_textures_region_"+litho[13]+"_v_join_fullProper.csv"

filePPP_0 = pd.read_csv(nameFileExp0,sep=",")
filePPP_1 = pd.read_csv(nameFileExp1,sep=",")
filePPP_2 = pd.read_csv(nameFileExp2,sep=",")
filePPP_3 = pd.read_csv(nameFileExp3,sep=",")

filePPP_5 = pd.read_csv(nameFileExp5,sep=",")
filePPP_6 = pd.read_csv(nameFileExp6,sep=",")

filePPP_8 = pd.read_csv(nameFileExp8,sep=",")
filePPP_9 = pd.read_csv(nameFileExp9,sep=",")
filePPP_10 = pd.read_csv(nameFileExp10,sep=",")
filePPP_11 = pd.read_csv(nameFileExp11,sep=",")
filePPP_12 = pd.read_csv(nameFileExp12,sep=",")
filePPP_13 = pd.read_csv(nameFileExp13,sep=",")

filePPP_0 = filePPP_0[filePPP_0["mean_offset"] > 0]
filePPP_1 = filePPP_1[filePPP_1["mean_offset"] > 0]
filePPP_2 = filePPP_2[filePPP_2["mean_offset"] > 0]
filePPP_3 = filePPP_3[filePPP_3["mean_offset"] > 0]

filePPP_5 = filePPP_5[filePPP_5["mean_offset"] > 0]
filePPP_6 = filePPP_6[filePPP_6["mean_offset"] > 0]

filePPP_8 = filePPP_8[filePPP_8["mean_offset"] > 0]
filePPP_9 = filePPP_9[filePPP_9["mean_offset"] > 0]
filePPP_10 = filePPP_10[filePPP_10["mean_offset"] > 0]
filePPP_11 = filePPP_11[filePPP_11["mean_offset"] > 0]
filePPP_12 = filePPP_12[filePPP_12["mean_offset"] > 0]
filePPP_13 = filePPP_13[filePPP_13["mean_offset"] > 0]


join_fileP = [filePPP_0 , filePPP_1 , filePPP_2 , filePPP_3 , filePPP_5 , filePPP_6 , filePPP_8, filePPP_9, filePPP_10, filePPP_11, filePPP_12, filePPP_13]
datasets_RF = config.datasets_RF + "dataset2/U1480/group1/"
name_file = datasets_RF+"dataset2_U1480_group1.csv"
result = pd.concat(join_fileP)

result.to_csv(name_file)
print ("Create dataset2-U1480-Group1") 

#group 2
for x in range(0,len(result)):
    
    if (result['cod_lit'][x] == 0) or (result['cod_lit'][x] == 1) or (result['cod_lit'][x] == 10) :
        result['cod_lit'][x] = 0
        
    if (result['cod_lit'][x] == 2):
        result['cod_lit'][x] = 1
        
    if (result['cod_lit'][x] == 3) or (result['cod_lit'][x] == 4):
        result['cod_lit'][x] = 2

    if (result['cod_lit'][x] == 5):
        result['cod_lit'][x] = 3
        
    if (result['cod_lit'][x] == 11) or (result['cod_lit'][x] == 12):
        result['cod_lit'][x] = 4
        
    if (result['cod_lit'][x] == 6) or (result['cod_lit'][x] == 8) or (result['cod_lit'][x] == 9) or (result['cod_lit'][x] == 13):
        result['cod_lit'][x] = 5

datasets_RF = config.datasets_RF + "dataset2/U1480/group2/"
name_file = datasets_RF+"dataset2_U1480_group2.csv"   
result.to_csv(name_file)          
print ("Create dataset2-U1480-Group2") 




#to dataset2-U1481


patch_out_description = config.directory_geo_prop_orig_interp_U1481

litho = config.litho

nameFileExp0 = patch_out_description +"extract_textures_region_"+litho[0]+"_v_join_fullProper.csv"
nameFileExp1 = patch_out_description +"extract_textures_region_"+litho[1]+"_v_join_fullProper.csv"
nameFileExp2 = patch_out_description +"extract_textures_region_"+litho[2]+"_v_join_fullProper.csv"

nameFileExp5 = patch_out_description +"extract_textures_region_"+litho[5]+"_v_join_fullProper.csv"

nameFileExp8 = patch_out_description +"extract_textures_region_"+litho[8]+"_v_join_fullProper.csv"
nameFileExp9 = patch_out_description +"extract_textures_region_"+litho[9]+"_v_join_fullProper.csv"

nameFileExp11 = patch_out_description +"extract_textures_region_"+litho[11]+"_v_join_fullProper.csv"
nameFileExp12 = patch_out_description +"extract_textures_region_"+litho[12]+"_v_join_fullProper.csv"
nameFileExp13 = patch_out_description +"extract_textures_region_"+litho[13]+"_v_join_fullProper.csv"

filePPP_0 = pd.read_csv(nameFileExp0,sep=",")
filePPP_1 = pd.read_csv(nameFileExp1,sep=",")
filePPP_2 = pd.read_csv(nameFileExp2,sep=",")

filePPP_5 = pd.read_csv(nameFileExp5,sep=",")

filePPP_8 = pd.read_csv(nameFileExp8,sep=",")
filePPP_9 = pd.read_csv(nameFileExp9,sep=",")
filePPP_11 = pd.read_csv(nameFileExp11,sep=",")
filePPP_12 = pd.read_csv(nameFileExp12,sep=",")
filePPP_13 = pd.read_csv(nameFileExp13,sep=",")

filePPP_0 = filePPP_0[filePPP_0["mean_offset"] > 0]
filePPP_1 = filePPP_1[filePPP_1["mean_offset"] > 0]
filePPP_2 = filePPP_2[filePPP_2["mean_offset"] > 0]

filePPP_5 = filePPP_5[filePPP_5["mean_offset"] > 0]

filePPP_8 = filePPP_8[filePPP_8["mean_offset"] > 0]
filePPP_9 = filePPP_9[filePPP_9["mean_offset"] > 0]
filePPP_11 = filePPP_11[filePPP_11["mean_offset"] > 0]
filePPP_12 = filePPP_12[filePPP_12["mean_offset"] > 0]
filePPP_13 = filePPP_13[filePPP_13["mean_offset"] > 0]


join_fileP = [filePPP_0 , filePPP_1 , filePPP_2 , filePPP_5 , filePPP_8, filePPP_9, filePPP_11, filePPP_12, filePPP_13]
datasets_RF = config.datasets_RF + "dataset2/U1481/group1/"
name_file = datasets_RF+"dataset2_U1481_group1.csv"
result = pd.concat(join_fileP)

result.to_csv(name_file)
print ("Create dataset2-U1481-Group1") 

#group 2
for x in range(0,len(result)):
    
    if (result['cod_lit'][x] == 0) or (result['cod_lit'][x] == 1) or (result['cod_lit'][x] == 10) :
        result['cod_lit'][x] = 0
        
    if (result['cod_lit'][x] == 2):
        result['cod_lit'][x] = 1
        
    if (result['cod_lit'][x] == 3) or (result['cod_lit'][x] == 4):
        result['cod_lit'][x] = 2

    if (result['cod_lit'][x] == 5):
        result['cod_lit'][x] = 3
        
    if (result['cod_lit'][x] == 11) or (result['cod_lit'][x] == 12):
        result['cod_lit'][x] = 4
        
    if (result['cod_lit'][x] == 6) or (result['cod_lit'][x] == 8) or (result['cod_lit'][x] == 9) or (result['cod_lit'][x] == 13):
        result['cod_lit'][x] = 5

datasets_RF = config.datasets_RF + "dataset2/U1481/group2/"
name_file = datasets_RF+"dataset2_U1481_group2.csv"   
result.to_csv(name_file)   
print ("Create dataset2-U1481-Group2") 





                