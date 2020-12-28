
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import csv
import sys
import config

np.set_printoptions(threshold=sys.maxsize)


datasets_RF = config.datasets_RF

dataset2_U1480_group1 = datasets_RF+"dataset2/U1480/group1/"
dataset2_U1480_group2 = datasets_RF+"dataset2/U1480/group2/"

dataset2_U1481_group1 = datasets_RF+"dataset2/U1481/group1/"
dataset2_U1481_group2 = datasets_RF+"dataset2/U1481/group2/"


#test_size_a = [0.10,0.20,0.30,0.40,0.50]
test_size_a = [0.2]
test_name = '20'
#train_size_a = [0.90,0.80,0.70,0.60,0.50]
train_size_a = [0.8]
train_name = '80'


#using
max_depth__ = np.arange(start=50, stop=51, step=1)
n_estimators__ = np.arange(start=100, stop=101, step=1)


#class
class_CM = ['0','1','3','4','5'] #adjust according to cod_lit and group
#class_CM = ['0','1','2','5','8','9','11','12','13'] #adjust according to cod_lit and group

#site
site = ['U1480'] #adjust            
            

acc  = []
acc1 = []

saveDir = config.saveDir  + "/dataset2/" + sitee + "/" 

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
    
    

filePPP = pd.read_csv("practical_arrangement_to-dataset2.csv",sep=",") # practical arrangement used according described in the article

for rr in range(0,len(filePPP)):
    variables      = filePPP['variables'][rr]
    id             = filePPP['id'][rr]
    group          = filePPP['group'][rr]
    
        
        
    if (site[0] == 'U1480'):    
        if (group == 1):
            name_file = dataset2_U1480_group1+"dataset2_U1480_group1.csv"
            groupFile = 'group1'
            groupNome = 'Group 1'
            print ("dataset2 U1480 ",groupFile)
        else:
            name_file = dataset2_U1480_group2+"dataset2_U1480_group2.csv"
            groupFile = 'group2'
            groupNome = 'Group 2'
            print ("dataset2 U1480 ",groupFile)

            
    else:
        if (group == 1):
            name_file = dataset2_U1481_group1+"dataset2_U1481_group1.csv"
            groupFile = 'group1'
            groupNome = 'Group 1'
            print ("dataset2 U1481 ",groupFile)
            
        else: 
            name_file = dataset2_U1481_group2+"dataset2_U1481_group2.csv"    
            groupFile = 'group2'
            groupNome = 'Group 2'   
            print ("dataset2 U1481 ",groupFile)            
        


    train = pd.read_csv(name_file,sep=",")
    

    L = variables.split("'")
    
    c = len (L)
    for i in range (0,c):
        for i in range (0,c):
            if (L[i] == ',' or L[i] == ''):
                del L[i]
                break
            
        c = len (L)

            
    column_names = L       
        

                          
    X = np.array(train[column_names].values)
    y = np.array(train['cod_lit'].values)

    X_trainN, X_testN, y_trainN, y_testN = train_test_split(X, y, test_size=test_size_a[0],random_state=42)


    X_trainnormalized = preprocessing.normalize(X, norm='max', axis=0)
           
    X_train, X_test, y_train, y_test = train_test_split(X_trainnormalized, y, test_size=test_size_a[0],random_state=42)


    #loop test in max_depth and n_estimators
    for i in range (0,len(max_depth__)):
        for ii in range (0,len(n_estimators__)):
        
            classifier_rf = RandomForestClassifier(max_depth=max_depth__[i], n_estimators=n_estimators__[ii], random_state=42, n_jobs=-1,class_weight='balanced')

            classifier_rf.fit(X_train, y_train)
                                
            y_pred = classifier_rf.predict(X_test)
                        
            m1 = metrics.classification_report(y_pred, y_test)                    
            #print(m1)
                                
            #Accuracy store
            #print ("Accuracy: ")
            m2 = metrics.accuracy_score(y_pred, y_test)
            #print(m2)

                
            print ("Report Accuracy: ")
            print_accuracy_report(classifier_rf, X_train, y_train)
            cross = print_accuracy_report(classifier_rf, X_train, y_train)
            

            df1 = pd.DataFrame(X_testN, columns = column_names) 
            df2 = pd.DataFrame(y_test, columns = ['cod_lit'])
            df3 = pd.DataFrame(y_pred, columns = ['cod_lit_test']) 


            result = pd.concat([df1, df3], axis=1, sort=False)
            
            
            namer = saveDir+"result_RF_texture_"+str(id)+"_"+str(group)+"_"+str(max_depth__[i])+"_"+str(n_estimators__[ii])+"_"+str(test_size_a[0])+".csv"
            result.to_csv(namer)  

            result = pd.concat([df1, df2, df3], axis=1, sort=False)
            namer_ = str(saveDir)+"result_RF_predict_"+str(id)+"_"+str(group)+"_"+str(max_depth__[i])+"_"+str(n_estimators__[ii])+"_"+str(test_size_a[0])+".csv"
            result.to_csv(namer_) 
            
               
            df22 = pd.read_csv(namer,sep=",")
            df11 = pd.read_csv(name_file,sep=",")



            result = pd.merge(df11, df22, how='left')
            #save result of test
            namerr = saveDir+"join_RF_texture_test_"+str(id)+"_"+str(group)+"_"+str(max_depth__[i])+"_"+str(n_estimators__[ii])+"_"+str(test_size_a[0])+".csv"
            result.to_csv(namerr)
            
            acc.append(group)
            acc.append(id)
            acc.append(max_depth__[i])
            acc.append(n_estimators__[ii])
            acc.append(m2)
            acc.append(cross)
            
            acc1.append(acc)
            acc = []
            

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
                plt.savefig( config.results_graphics +"dataset2/ConfMatrix_"+str(groupFile)+"_"+str(test_size_a[0])+".png",dpi=96)
                       


            cnf_matrix = confusion_matrix(y_test, y_pred)
            np.set_printoptions(precision=2)

            class_names = class_CM #ajust cod_lit

            # Plot non-normalized confusion matrix
            plt.figure(figsize=(8, 6))



            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                  title='Confusion matrix ('+str(train_name)+'% Train, '+str(test_name)+'% Test), '+str(groupNome))
    
    

    cols = ['group','id','max_depth','n_estimators','Accuracy','Cross']
    dt_result = pd.DataFrame(acc1,columns=cols)

    nameFileExp1 = saveDir+'result_RF_texture_acc_'+"_"+str(id)+"_"+str(group)+"_"+str(test_size_a[0])+".csv"

    dt_result.to_csv(nameFileExp1)
    acc1=[]
    acc = []
    print ("Line ",id,"completed.")

 
