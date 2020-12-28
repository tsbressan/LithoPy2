
import pandas as pd 
import numpy as np
import os
import csv
import json
import config

pd.options.display.width = 0

patch_properties = config.directory_geo_prop_orig_interp_U1480 + "dataset1/"
patch_output     = config.directory_output_images_U1480 


prop = ['gra','mad','ms','ngr','pwl','rgb','rsc']



#read files in directory
files_properties = []
for filename in os.listdir(patch_properties):
    files_properties.append (filename)

files_output_litho=[]    
for filename1 in os.listdir(patch_output):
    files_output_litho.append (filename1)




def export_interp_and_csv(data,name_csv_export):
    column_names = ["Exp", "Site", "Hole","Core","Type","Sect","A/W","Offset","value_1_gra","value_1_mad",
                    "value_2_mad","value_3_mad","value_4_mad","value_5_mad","value_6_mad","value_1_ms",
                    "value_1_ngr","value_1_pwl","value_1_rgb","value_2_rgb","value_3_rgb","value_1_rsc",
                    "value_2_rsc","value_3_rsc","lithology"]
    
    df = pd.DataFrame(data,columns=column_names)

    
    df.sort_values(by=['Exp','Site','Hole','Core','Type','Sect','Offset'], inplace=True)
    df1 =  df.drop_duplicates() 

    
    aggregation_functions = {'value_1_gra': 'sum', 'value_1_mad': 'sum', 'value_2_mad': 'sum', 'value_3_mad': 'sum', 
                             'value_4_mad': 'sum', 'value_5_mad': 'sum', 'value_6_mad': 'sum', 'value_1_ms': 'sum', 
                             'value_1_ngr': 'sum', 'value_1_pwl': 'sum', 'value_1_rgb': 'sum', 'value_2_rgb': 'sum', 
                             'value_3_rgb': 'sum', 'value_1_rsc': 'sum', 'value_2_rsc': 'sum', 'value_3_rsc': 'sum',
                             'lithology':'max'}

    df2 = df1.groupby(['Exp', 'Site','Hole','Core','Type','Sect','Offset']).aggregate(aggregation_functions)
    
    
    #save dataframe in .csv
    name_file_ = patch_properties+name_csv_export+"_interp_temp.csv"
    df2.to_csv(name_file_)
                  
        
     #import .csv to interpolation
    df3 = pd.read_csv(name_file_,sep=",")

    
    #adjust field value
    cols_alter = ["value_1_gra","value_1_mad","value_2_mad","value_3_mad","value_4_mad","value_5_mad","value_6_mad","value_1_ms","value_1_ngr","value_1_pwl","value_1_rgb","value_2_rgb","value_3_rgb","value_1_rsc","value_2_rsc","value_3_rsc"]
    for fc in range(0,len(df3)):
    
        for fcc in range(0,len(cols_alter)):
            #if (cols_alter[fcc] != 'value_3_rsc') and (df3["Hole"][fc] != 'g') and (df3["Core"][fc] != 45):
            
            if ( ( (cols_alter[fcc] == 'value_3_rsc') and (df3["Hole"][fc] == 'g') and (df3["Core"][fc] == 45) ) or 
                      ( (cols_alter[fcc] == 'value_1_ms') and (df3["Hole"][fc] == 'g') and (df3["Core"][fc] == 46) ) or
                      ( (cols_alter[fcc] == 'value_1_rgb') and (df3["Hole"][fc] == 'g') and (df3["Core"][fc] == 50) ) or
                      ( ( (cols_alter[fcc] == 'value_1_gra') or (cols_alter[fcc] == 'value_1_ms') ) and (df3["Hole"][fc] == 'f') and (df3["Core"][fc] == 6) ) or 
                      ( ( (cols_alter[fcc] == 'value_1_gra') or (cols_alter[fcc] == 'value_1_ms') ) and (df3["Hole"][fc] == 'f') and (df3["Core"][fc] == 12) ) or 
                      ( ( (cols_alter[fcc] == 'value_1_gra') or (cols_alter[fcc] == 'value_1_ms') ) and (df3["Hole"][fc] == 'f') and (df3["Core"][fc] == 33) ) or
                      ( ( (cols_alter[fcc] == 'value_1_gra') or (cols_alter[fcc] == 'value_1_ms') ) and (df3["Hole"][fc] == 'f') and (df3["Core"][fc] == 53) ) ):
                      
            #if ( (df3["Hole"][fc] == 'g') and (df3["Core"][fc] == 50) and (cols_alter[fcc] == 'value_1_rgb') ):
                pass
            else:
                value_c       = str(df3[cols_alter[fcc]][fc])
                value_ = value_c.split('.')
                if (len(value_) > 2):
                    v2 = value_[1]
                    new_value = value_[0] + "." + v2[0:1]
                    df3[cols_alter[fcc]][fc] = float(new_value)
            
    df3.to_csv(name_file_)
    
    df3 = pd.read_csv(name_file_,sep=",")
    

    
    #test count element in properties   
    test_count = []
    count_df3 = df3.count()    
    
    for fii in range(0,len(count_df3)):
        if (count_df3[fii]>2):
            test_count.append("ok")
        else:
            test_count.append("no")
        
    
    print (len(df3))

    name_file1 = patch_properties+"full_interp_csv"+"_"+name_csv_export
    
    
    #execut interpolation and save in .csv
    
    method_inter = ["akima"]
        
    if ('no' in test_count):
        print ("Error - record <= 2")
        del(df3)
        del(test_count)
        del(method_inter)
        del(data)
    else:

        for u in range (0,len(method_inter)):                               

            df_import2 = df3.interpolate(method=method_inter[u], limit_direction ='both')
            name_file = name_file1+"_"+method_inter[u] 


            name_file = name_file+".csv"
            df_import2.to_csv(name_file)
            
            df_import22 = pd.read_csv(name_file,sep=",")
            
            cols_alter = ["value_1_gra","value_1_mad","value_2_mad","value_3_mad","value_4_mad","value_5_mad","value_6_mad","value_1_ms","value_1_ngr","value_1_pwl","value_1_rgb","value_2_rgb","value_3_rgb","value_1_rsc","value_2_rsc","value_3_rsc"]
    
            for fc in range(0,len(df_import22)):
            
                for fcc in range(0,len(cols_alter)):
                    value_c       = str(df_import22[cols_alter[fcc]][fc])
                    value_ = value_c.split('.')
                    if (len(value_) > 2):
                        v2 = value_[1]
                        new_value = value_[0] + "." + v2[0:1]
                        df_import22[cols_alter[fcc]][fc] = float(new_value)                
            
            
            df_import22.to_csv(name_file)
            name_file = ''
            
            
exp = ['362']
site = ['U1480']
hole = ['F'] #run one by one

#core = np.arange(start=55, stop=56, step=1)
core = np.arange(start=56, stop=100, step=1) #adjust according to hole interval - core not used: E-8, F-55, G-21,60
typee = ['H','F','X', 'R']

sect = ['1','2','3','4','5','6','7','8','9','cc']

region = np.arange(start=0, stop=41, step=1)


log=''
log1=''
log_offset = ''
data=[]
method_inter=[]
contFilecsv=1

for x in range(0,len(exp)):
    
    for xx in range(0,len(site)):
        for xxx in range(0,len(hole)):
            for xxxx in range(0,len(core)):
                for yy in range(0,len(typee)):
                
                    for xxxxx in range(0,len(sect)):
                        for y in range(0,len(region)):
                            for FnLitho in range (0,len(files_output_litho)):   


                                site_str = str(site[xx])
                                hole_str = str(hole[xxx])
                                typee_str= str(typee[yy])

                                
                                nome = str(exp[x])+"-"+site_str.lower()+hole_str.lower()+"-"+str(core[xxxx])+typee_str.lower()+"-"+str(sect[xxxxx])+"-a;region-"+str(region[y])+";"+str(files_output_litho[FnLitho])+".csv"
                                #print (nome)

                                for fp in range (0,len(files_properties)):

                                    if (files_properties[fp] == nome) and (log != nome):
                                        filePPP = pd.read_csv(patch_properties+files_properties[fp],sep=",")
                                        log=nome
                                        filePPP.drop_duplicates(subset=['Offset'])
                                        
                                        #print (nome)

                                        file_json_csv = pd.read_csv(config.file_json_csv_U1480,sep=",")
                                        for Fj in range(0,len(file_json_csv)):
                                            filename_json_csv      = file_json_csv['filename'][Fj]
                                            region_id_json_csv     = file_json_csv['region_id'][Fj]


                                            split_filename_csv     = filename_json_csv.split('_')
                                            name_full_image        = str(exp[x])+'-'+site_str.lower()+hole_str.lower()+'-'+str(core[xxxx])+typee_str.lower()+'-'+str(sect[xxxxx])+'-a'

                                            if (split_filename_csv[0] == name_full_image) and (log1 != name_full_image):

                                                regAtt_json_csv      = file_json_csv['region_attributes'][Fj]
                                                
                                                split_regAtt_csv     = regAtt_json_csv.split('Depth')
                                                split1               = split_regAtt_csv[1]
                                                split1_              = split1.split('"')

                                                depth_ma_me          = split1_[2]
                                                split_depth_ma_me    = depth_ma_me.split('_')
                                                min_depth            = float(split_depth_ma_me[0])
                                                max_depth            = float(split_depth_ma_me[1])

                                                log1=name_full_image

                                        #print (name_full_image)
                                        #print (min_depth)
                                        #print (max_depth)

                                        offset_ = np.arange(min_depth,max_depth+1,1)
                                        if (len(filePPP) > 0):
                                            for fii in range(0,len(filePPP)):
                                                Lithology__i   = str(filePPP['Lithology'][fii])
                                                Prop__i        = str(filePPP['Prop'][fii])
                                                value__i       = filePPP['value'][fii]
                                                Exp__i         = filePPP['Exp'][fii]
                                                SiteHole__i    = filePPP['SiteHole'][fii]
                                                CoreType__i    = filePPP['CoreType'][fii]
                                                Sect__i        = filePPP['Sect'][fii]

                                                Offset__i      = filePPP['Offset'][fii]


                                                Site__i = SiteHole__i[0:5]
                                                Hole__i = SiteHole__i[5:6]
                                                if (len(CoreType__i)) == 2:
                                                    Core__i = CoreType__i[0:1]
                                                    Type__i = CoreType__i[1:2]
                                                elif (len(CoreType__i)) == 3:
                                                    Core__i = CoreType__i[0:2]
                                                    Type__i = CoreType__i[2:3]


                                                for R in range(0,len(offset_)):                                                    
                                                    if (Prop__i == 'gra'):   
                                                        if (Offset__i != offset_[R]):
                                                            data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',offset_[R],'','','','','','','','','','','','','','','','',''])

                                                        data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',Offset__i,value__i,'','','','','','','','','','','','','','','',Lithology__i])
                                                        #data.extend - test

                                                    if (Prop__i == 'mad'):
                                                        if (Offset__i != offset_[R]):
                                                            data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',offset_[R],'','','','','','','','','','','','','','','','',''])

                                                        mad_value = value__i.split('_')
                                                        Moisturedry  = mad_value[0]
                                                        Moisturewet  = mad_value[1]
                                                        Bulkdensity  = mad_value[2]
                                                        Drydensity   = mad_value[3]
                                                        Graindensity = mad_value[4]
                                                        Porosity     = mad_value[5]

                                                        data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',Offset__i,'',Moisturedry,Moisturewet,Bulkdensity,Drydensity,Graindensity,Porosity,'','','','','','','','','',Lithology__i])


                                                    if (Prop__i == 'ms'):
                                                        if (Offset__i != offset_[R]):
                                                            data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',offset_[R],'','','','','','','','','','','','','','','','',''])

                                                        Magneticsusceptibility = value__i
                                                        data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',Offset__i,'','','','','','','',Magneticsusceptibility,'','','','','','','','',Lithology__i])


                                                    if (Prop__i == 'ngr'):
                                                        if (Offset__i != offset_[R]):
                                                            data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',offset_[R],'','','','','','','','','','','','','','','','',''])

                                                        NGRtotalcounts = value__i
                                                        data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',Offset__i,'','','','','','','','',NGRtotalcounts,'','','','','','','',Lithology__i])

                                                    if (Prop__i == 'pwl'):
                                                        if (Offset__i != offset_[R]):
                                                            data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',offset_[R],'','','','','','','','','','','','','','','','',''])

                                                        Pwavevelocity = value__i
                                                        data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',Offset__i,'','','','','','','','','',Pwavevelocity,'','','','','','',Lithology__i])

                                                    if (Prop__i == 'rgb'):
                                                        if (Offset__i != offset_[R]):
                                                            data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',offset_[R],'','','','','','','','','','','','','','','','',''])

                                                        rgb_value = value__i.split('_')
                                                        R = rgb_value[0]
                                                        G = rgb_value[1]
                                                        B = rgb_value[2]
                                                        data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',Offset__i,'','','','','','','','','','',R,G,B,'','','',Lithology__i])

                                                    if (Prop__i == 'rsc'):
                                                        if (Offset__i != offset_[R]):
                                                            data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',offset_[R],'','','','','','','','','','','','','','','','',''])

                                                        rsc_value = value__i.split('_')
                                                        ReflectanceL = rsc_value[0]
                                                        Reflectancea = rsc_value[1]
                                                        Reflectanceb = rsc_value[2]
                                                        data.append([Exp__i,Site__i,Hole__i,Core__i,Type__i,Sect__i,'',Offset__i,'','','','','','','','','','','','','',ReflectanceL,Reflectancea,Reflectanceb,Lithology__i])


                    if (len(data) > 0):                    
                        name_csv_export = str(exp[x])+'_'+str(site[xx])+'_'+str(hole[xxx])+'_'+str(core[xxxx])+''+str(typee[yy])                   
                        export_interp_and_csv(data,name_csv_export)
                        del(data)
                        print (name_csv_export)
                        data=[]
                    else:
                        del(data)
                        data=[]

   