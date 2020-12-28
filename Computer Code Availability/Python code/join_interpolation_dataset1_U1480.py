
import pandas as pd 
import numpy as np
import os
import csv
import json
import config

pd.options.display.width = 0


patch_properties = config.directory_geo_prop_orig_interp_U1480 + "dataset1/"
patch_output     = config.directory_output_images_U1480 


#read files in directory
files_properties = []
for filename in os.listdir(patch_properties):
    files_properties.append (filename)

  
  

method_inter = ["akima"]   


exp = ['362']
site = ['U1480']
hole = ['F','G','H'] #run all or run one by one

core = np.arange(start=1, stop=100, step=1) #adjust according to hole interval

typee = ['H','F','X', 'R']
sect = ['1','2','3','4','5','6','7','8','9','cc']


log=''
df_full = []  
lithology__i_temp = ''
lithology__i_cod = ''

files_output_litho=[]    
for filename1 in os.listdir(patch_output):
    files_output_litho.append (filename1)
    

for y in range(0,len(method_inter)): 
    for x in range(0,len(exp)):
        for xx in range(0,len(site)):
            for xxx in range(0,len(hole)):
                for yy in range(0,len(typee)):
                    for xxxx in range(0,len(core)):
                        for xxxxx in range(0,len(sect)):

                            site_str = str(site[xx])
                            hole_str = str(hole[xxx])
                            core_str = str(core[xxxx])
                            sect_str = str(sect[xxxxx])
                            typee_str= str(typee[yy])    


                            nome = 'full_interp_csv_'+str(exp[x])+"_"+site_str+'_'+hole_str+"_"+core_str+typee_str+"_"+str(method_inter[y])+".csv"
                            name_file = patch_properties+str(method_inter[y])+'_full_'+str(exp[x])+"_"+site_str+'_'+hole_str+".csv"

                                
                            for fp in range (0,len(files_properties)):
                                if (files_properties[fp] == nome) and (log!=nome):
                                    print (nome)
                                    log=nome
                                    filePPP = pd.read_csv(patch_properties+files_properties[fp],sep=",")
                                    
                                    #detail the fields
                                    for fii in range(0,len(filePPP)):


                                        Exp__i         = filePPP['Exp'][fii]
                                        Site__i        = filePPP['Site'][fii]
                                        Hole__i        = filePPP['Hole'][fii]
                                        Core__i        = filePPP['Core'][fii]
                                        Type__i        = filePPP['Type'][fii]
                                        Sect__i        = filePPP['Sect'][fii]
                                        Offset__i      = filePPP['Offset'][fii]

                                        #GRA
                                        value_1_gra__i_ = str(filePPP['value_1_gra'][fii])
                                        split_gra1 = value_1_gra__i_.split('.')
                                        if (len(split_gra1) == 3):
                                            value_1_gra__i = round(float(split_gra1[0]+'.'+split_gra1[1]),2)
                                        else:
                                            value_1_gra__i = round(float(value_1_gra__i_),2)


                                        value_1_mad__i = round(float(filePPP['value_1_mad'][fii]),2)
                                        value_2_mad__i = round(float(filePPP['value_2_mad'][fii]),2)
                                        value_3_mad__i = round(float(filePPP['value_3_mad'][fii]),2)
                                        value_4_mad__i = round(float(filePPP['value_4_mad'][fii]),2)
                                        value_5_mad__i = round(float(filePPP['value_5_mad'][fii]),2)
                                        value_6_mad__i = round(float(filePPP['value_6_mad'][fii]),2)

                                        #ms
                                        value_1_ms__i_ = str(filePPP['value_1_ms'][fii])
                                        split_ms1 = value_1_ms__i_.split('.')
                                        if (len(split_ms1) == 3):
                                            value_1_ms__i = round(float(split_ms1[0]+'.'+split_ms1[1]),2)
                                        else:
                                            value_1_ms__i = round(float(value_1_ms__i_),2)                                   



                                        value_1_ngr__i = round(float(filePPP['value_1_ngr'][fii]),2)
                                        value_1_pwl__i = round(float(filePPP['value_1_pwl'][fii]),2)


                                        #RGB - R
                                        value_1_rgb__i_ = str(filePPP['value_1_rgb'][fii])
                                        split_rgb1 = value_1_rgb__i_.split('.')
                                        if (len(split_rgb1) == 3):
                                            value_1_rgb__i = round(float(split_rgb1[0]+'.'+split_rgb1[1]),2)
                                        else:
                                            value_1_rgb__i = round(float(value_1_rgb__i_),2)

                                        #RGB - G
                                        value_2_rgb__i_ = str(filePPP['value_2_rgb'][fii])
                                        split_rgb2 = value_2_rgb__i_.split('.')
                                        if (len(split_rgb2) == 3):
                                            value_2_rgb__i = round(float(split_rgb2[0]+'.'+split_rgb2[1]),2)
                                        else:
                                            value_2_rgb__i = round(float(value_2_rgb__i_),2) 

                                        #RGB - B    
                                        value_3_rgb__i_ = str(filePPP['value_3_rgb'][fii])
                                        split_rgb3 = value_3_rgb__i_.split('.')
                                        if (len(split_rgb3) == 3):
                                            value_3_rgb__i = round(float(split_rgb3[0]+'.'+split_rgb3[1]),2)
                                        else:
                                            value_3_rgb__i = round(float(value_3_rgb__i_),2) 

                                        #RSC1    
                                        value_1_rsc__i_ = str(filePPP['value_1_rsc'][fii])
                                        split_rsc1 = value_1_rsc__i_.split('.')
                                        if (len(split_rsc1) == 3):
                                            value_1_rsc__i = round(float(split_rsc1[0]+'.'+split_rsc1[1]),2)
                                        else:
                                            value_1_rsc__i = round(float(value_1_rsc__i_),2) 

                                        #RSC2                      
                                        value_2_rsc__i_ = str(filePPP['value_2_rsc'][fii])
                                        split_rsc2 = value_2_rsc__i_.split('.')
                                        if (len(split_rsc2) == 3):
                                            value_2_rsc__i = round(float(split_rsc2[0]+'.'+split_rsc2[1]),2)
                                        else:
                                            value_2_rsc__i = round(float(value_2_rsc__i_),2) 

                                        #RSC3                      
                                        value_3_rsc__i_ = str(filePPP['value_3_rsc'][fii])
                                        split_rsc3 = value_3_rsc__i_.split('.')
                                        if (len(split_rsc3) == 3):
                                            split__ = split_rsc3[1]
                                            split_ = split__[0:2]
                                            value_3_rsc__i = round(float(split_rsc3[0]+'.'+split_),2)
                                        else:
                                            value_3_rsc__i = round(float(value_3_rsc__i_),2) 


                                        lithology__i   = filePPP['lithology'][fii]



                                        #add cod_lit
                                        for fcod in range (0,len(files_output_litho)):
                                            if (files_output_litho[fcod] == lithology__i):
                                                cod_lit = fcod
                                                lithology__i_temp = lithology__i
                                                lithology__i_cod = fcod
                                                
                                                
                                        
                                        if pd.isnull(lithology__i):
                                            cod_lit = lithology__i_cod
                                            lithology__i = lithology__i_temp


                                        df_full.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,Offset__i,value_1_gra__i,value_1_mad__i,
                                                    value_2_mad__i,value_3_mad__i,value_4_mad__i,value_5_mad__i,value_6_mad__i,value_1_ms__i,value_1_ngr__i
                                                    ,value_1_pwl__i,value_1_rgb__i,value_2_rgb__i,value_3_rgb__i,value_1_rsc__i,value_2_rsc__i,value_3_rsc__i,
                                                    lithology__i,cod_lit])
                                    
                                    
                                
    
                print (method_inter[y])
                
                column_names = ["Exp", "Site", "Hole","Core","Type","Sect","Offset","value_1_gra","value_1_mad",
                                "value_2_mad","value_3_mad","value_4_mad","value_5_mad","value_6_mad","value_1_ms",
                                "value_1_ngr","value_1_pwl","value_1_rgb","value_2_rgb","value_3_rgb","value_1_rsc",
                                "value_2_rsc","value_3_rsc","lithology","cod_lit"]
                
                df_full1 = pd.DataFrame(df_full,columns=column_names)
                
                #empty data dataframe
                df_full1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
                print (len (df_full1))
                #export .csv
                df_full1.to_csv(name_file)
                
                #reset variables
                df_full  = []
                df_full1 = []


