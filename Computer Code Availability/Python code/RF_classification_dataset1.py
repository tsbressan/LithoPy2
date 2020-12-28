
import itertools
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import csv


import warnings
warnings.filterwarnings("ignore")
import config




datasets_RF = config.datasets_RF

dataset1_U1480_group1 = datasets_RF+"dataset1/U1480/group1/"
dataset1_U1480_group2 = datasets_RF+"dataset1/U1480/group2/"

dataset1_U1481_group1 = datasets_RF+"dataset1/U1481/group1/"
dataset1_U1481_group2 = datasets_RF+"dataset1/U1481/group2/"


exp = ['362']
site = ['U1480'] #adjust - U1480 or U1481
hole = ['E','F','G','H'] #adjust


#test_size_a = [0.10,0.20,0.30,0.40,0.50]
test_size_a = [0.10] #can run separately or the 4 combinations together.
test_name = '10' #name for title image
#train_size_a = [0.90,0.80,0.70,0.60,0.50]
train_size_a = [0.90] #can run separately or the 4 combinations together.
train_name = '90' #name for title image

RF_1 = 0
RF_2 = 2 
RF_3 = 1

#interval between number of trees and maximum of trees - adjust as you wish to execute.
max_depth__ = np.arange(start=50, stop=51, step=1)
n_estimators__ = np.arange(start=10, stop=101, step=10)

#group
groupFile = 'group2' #adjust
groupNome = 'Group 2' #name for title image

#class
class_CM = ['0','1','3','4','5'] #adjust according to cod_lit and group
#class_CM = ['0','1','2','3','5','8','9','11','12','13'] #adjust according to cod_lit and group



#save logs
saveDir = config.saveDir + "dataset1/" +site[0] + "/" 


def print_accuracy_report(classifier_rf, X_train, y_train, num_validations=5):
    accuracy = cross_val_score(classifier_rf, X_train, y_train, scoring='accuracy', cv=num_validations)
    #print ("Accuracy:" + str(round(100*accuracy.mean(), 2)) + "%")
    f1 = cross_val_score(classifier_rf,X_train, y_train, scoring='f1_macro', cv=num_validations)
    #print ("F1: " + str(round(100*f1.mean(), 2)) + "%")
    precision = cross_val_score(classifier_rf,X_train, y_train, scoring='precision_weighted', cv=num_validations)
    #print ("Precision: " + str(round(100*precision.mean(), 2)) + "%")
    recall = cross_val_score(classifier_rf, X_train, y_train, scoring='recall_weighted', cv=num_validations)
    #print ("Recall: " + str(round(100*recall.mean(), 2)) + "%")
    return accuracy.mean(), f1.mean(), precision.mean(), recall.mean()






#code randomForest
for xx in range(0,len(test_size_a)):
                
                
    print ("Test: ", test_size_a[xx]," --- Train: ", train_size_a[xx])
    print ("n_estimators: ", n_estimators__[xx]," --- max_depth__: ", max_depth__[xx])
                
    #-----------------------------------------------------------------------------------
    #phisical properties full with interpolador.
    if (RF_1 == 0):

            
        if (site[0] == 'U1480'):    
            if (groupFile == 'group1'):
                print ("akima_U1480_",groupFile)
                name_file = dataset1_U1480_group1+"dataset1_U1480_group1.csv"
            else:
                print ("akima_U1480_",groupFile)
                name_file = dataset1_U1480_group2+"dataset1_U1480_group2.csv"
                
        else:
            if (groupFile == 'group1'):
                print ("akima_U1481_",groupFile)
                name_file = dataset1_U1481_group1+"dataset1_U1481_group1.csv"
            else:
                print ("akima_U1481_",groupFile)
                name_file = dataset1_U1481_group2+"dataset1_U1481_group2.csv"            

                    
        train = pd.read_csv(name_file,sep=",")
        features = train.columns.difference(["id","Exp", "Site", "Hole","Core","Type","Sect","lithology","cod_lit"])
        column_names = ["Offset","value_1_gra","value_1_mad","value_2_mad","value_3_mad","value_4_mad","value_5_mad",
                                    "value_6_mad","value_1_ms","value_1_ngr","value_1_pwl","value_1_rgb","value_2_rgb",
                                    "value_3_rgb","value_1_rsc","value_2_rsc","value_3_rsc"]
        X = np.array(train[features].values)
        y = np.array(train['cod_lit'].values)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_a[xx],random_state=42)
        
        X_trainN, X_testN, y_trainN, y_testN = train_test_split(X, y, test_size=test_size_a[xx],random_state=42)
        
        #loop test in max_depth and n_estimators
        for i in range (0,len(max_depth__)):
        
        
            name_fi = str(saveDir) + "result_RF1_"+str(site[0])+"_"+str(groupFile)+"_"+str(test_size_a[0])+"_"+ str(max_depth__[i])+".csv"   
            with open(name_fi, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["max_depth", "n_estimators", "Accuracy","Accuracy Cross", "f1", "precision", "recall"])     
                
                
            for ii in range (0,len(n_estimators__)):
                classifier_rf = RandomForestClassifier(warm_start=True, oob_score=True, max_depth=max_depth__[i], n_estimators=n_estimators__[ii], random_state=42, n_jobs=-1,class_weight='balanced')
                            
                X_trainnormalized = preprocessing.normalize(X_train, norm='max', axis=0)
                X_testnormalized = preprocessing.normalize(X_test, norm='max', axis=0)
                            
                classifier_rf.fit(X_trainnormalized, y_train)
                                    
                y_pred = classifier_rf.predict(X_testnormalized)
                            
                m1 = metrics.classification_report(y_pred, y_test)                    
                #print(m1)
                                    
                #Accuracy store
                #print ("Accuracy: ")
                m2 = metrics.accuracy_score(y_pred, y_test)
                #print(m2)
                
                             
                                           
               
                df1 = pd.DataFrame(X_testN, columns = features) 
                df2 = pd.DataFrame(y_test, columns = ['cod_lit'])
                df3 = pd.DataFrame(y_pred, columns = ['cod_lit_test'])
                result = pd.concat([df1, df2, df3], axis=1, sort=False)
                namer = str(saveDir)+"result_RF_predict_"+str(groupFile)+"_"+str(test_size_a[xx])+"_"+str(max_depth__[xx])+"_"+str(n_estimators__[xx])+".csv"
                result.to_csv(namer) 
                
                
                df22 = pd.read_csv(namer,sep=",")
                df11 = pd.read_csv(name_file,sep=",")
                result = pd.merge(df11, df22, how='left')
                namerr = str(saveDir)+"join_RF_predict_test_"+str(groupFile)+"_"+str(test_size_a[xx])+"_"+str(max_depth__[xx])+"_"+str(n_estimators__[xx])+".csv"
                result.to_csv(namerr)
                               
                                    
                #create confusion matrix

                def plot_confusion_matrix(cm, classes,
                                          normalize=False,
                                          title='Confusion matrix',
                                          cmap=plt.cm.Blues
                                          ):
                    """
                    This function prints and plots the confusion matrix.
                    Normalization can be applied by setting `normalize=True`.
                    """
                    if normalize:
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        print("Normalized confusion matrix")
                    else:
                        print('Confusion matrix, without normalization')

                    print(cm)
                    
                    
                    plt.figure(figsize=(8, 6))
                    plt.imshow(cm, interpolation='nearest', cmap=cmap)
                    plt.title(title)
                    plt.colorbar()
                    tick_marks = np.arange(len(classes))
                    plt.xticks(tick_marks, classes, rotation=45)
                    plt.yticks(tick_marks, classes)

                    fmt = '.2f' if normalize else 'd'
                    thresh = cm.max() / 2.
                    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                        plt.text(j, i, format(cm[i, j], fmt),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")
                                 
                    len_class = float(len(class_CM))
                    len_class = len_class - 0.5
                    plt.xlim(-0.5, len_class) #figure size according to quantity of cod_lit
                    plt.ylim(len_class, -0.5) #figure size according to quantity of cod_lit

                    
                    
                    plt.tight_layout()
                    plt.ylabel('Predicted label')
                    plt.xlabel('True label')
                    plt.savefig(config.results_graphics + "dataset1/ConfMatrix_"+str(groupFile)+"_"+str(test_size_a[xx])+"_"+str(max_depth__[xx])+"_"+str(n_estimators__[xx])+".png",dpi=96)
                    
                    
                # Compute confusion matrix

                cnf_matrix = confusion_matrix(y_test, y_pred)
                np.set_printoptions(precision=2)

                class_names = class_CM #ajust cod_lit

                # Plot non-normalized confusion matrix
                plt.figure(figsize=(8, 6))



                plot_confusion_matrix(cnf_matrix, classes=class_names,
                                      #title='Confusion matrix, without normalization')
                                      title='Confusion matrix ('+str(train_name)+'% Train, '+str(test_name)+'% Test), '+str(groupNome))




                cross = print_accuracy_report(classifier_rf, X_train, y_train)
                    
                print ("max_depth:", max_depth__[i],"n_estimators:", n_estimators__[ii])
                
                
                #save results
                with open(name_fi, 'a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([max_depth__[i], n_estimators__[ii], m2, cross])
                

    #-----------------------------------------------------------------------------------

    
                
                

