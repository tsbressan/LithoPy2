
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import plot_confusion_matrix

import warnings
warnings.filterwarnings("ignore")

import config



datasets_RF = config.datasets_RF + "dataset0/"

#methods polynomial_order_1,polynomial_order_2,polynomial_order_3,from_derivatives not used, but the interpolation file is created.
method_inter = ["linear","polynomial_order_1","polynomial_order_2","polynomial_order_3","from_derivatives","spline_order_1","spline_order_2","spline_order_3","spline_order_4","slinear","quadratic","cubic","akima","pchip","piecewise_polynomial"]   


                
exp = ['362']
site = ['U1480']
hole = ['E']



def print_accuracy_report(classifier_rf, X_train, y_train, num_validations=5):
    accuracy = cross_val_score(classifier_rf, X_train, y_train, scoring='accuracy', cv=num_validations)
    print ("Accuracy:" + str(round(100*accuracy.mean(), 2)) + "%")
    f1 = cross_val_score(classifier_rf,X_train, y_train, scoring='f1_macro', cv=num_validations)
    print ("F1: " + str(round(100*f1.mean(), 2)) + "%")
    precision = cross_val_score(classifier_rf,X_train, y_train, scoring='precision_weighted', cv=num_validations)
    print ("Precision: " + str(round(100*precision.mean(), 2)) + "%")
    recall = cross_val_score(classifier_rf, X_train, y_train, scoring='recall_weighted', cv=num_validations)
    print ("Recall: " + str(round(100*recall.mean(), 2)) + "%") 
    
   
   
    
for u in range (0,len(method_inter)):
    for x in range(0,len(exp)):
        for xx in range(0,len(site)):
            for xxx in range(0,len(hole)):
                        
                name_file = datasets_RF+method_inter[u]+"_full_"+str(exp[x])+"_"+str(site[xx])+"_"+str(hole[xxx])+".csv"
                print (name_file)            
                        
                train = pd.read_csv(name_file,sep=",")
                #alter header - add column id
                new_header = ["id","Exp","Site","Hole","Core","Type","Sect","Offset","value_1_gra","value_1_mad","value_2_mad","value_3_mad","value_4_mad","value_5_mad","value_6_mad","value_1_ms","value_1_ngr","value_1_pwl","value_1_rgb","value_2_rgb","value_3_rgb","value_1_rsc","value_2_rsc","value_3_rsc","lithology","cod_lit"]
                train.to_csv(name_file, header=new_header, index=False)
                
                train = pd.read_csv(name_file,sep=",")
                
                        
                features = train.columns.difference(["id","Exp", "Site", "Hole","Core","Type","Sect","lithology","cod_lit"])

                column_names = ["Offset","value_1_gra","value_1_mad","value_2_mad","value_3_mad","value_4_mad","value_5_mad",
                                "value_6_mad","value_1_ms","value_1_ngr","value_1_pwl","value_1_rgb","value_2_rgb",
                                "value_3_rgb","value_1_rsc","value_2_rsc","value_3_rsc"]
                

                X = np.array(train[features].values)
                y = np.array(train['cod_lit'].values)
                        
                #adjust training combination: 0.10 or 0.20 or 0.30 or 0.40        
                X_train, X_test, y_train, y_test     = train_test_split(X, y, test_size=0.40,random_state=42)
                X_trainN, X_testN, y_trainN, y_testN = train_test_split(X, y, test_size=0.40,random_state=42)

                classifier_rf = RandomForestClassifier(max_depth=20, n_estimators=1000, random_state=42, n_jobs=-1)

                X_trainnormalized = preprocessing.normalize(X_train, norm='max', axis=0)
                X_testnormalized = preprocessing.normalize(X_test, norm='max', axis=0)

                #y_train is labels
                classifier_rf.fit(X_trainnormalized, y_train)
                        
                y_pred = classifier_rf.predict(X_testnormalized)
                        
                        
                        
                print(metrics.classification_report(y_pred, y_test))
                        
                #Accuracy store - ok
                print ("Accuracy: ")
                print(metrics.accuracy_score(y_pred, y_test))
                        
                #coefficient of pearson - not used
                #print ("coefficient of pearson: ")
                #print(metrics.r2_score(y_test, y_pred)) 
                        
                      

                print ("Report Accuracy: ")
                print_accuracy_report(classifier_rf, X_train, y_train)
    
                #feature importance.
                feature_importances = pd.DataFrame(classifier_rf.feature_importances_,index = column_names,
                                                   columns=['importance']).sort_values('importance', ascending=False)    
    
                print (feature_importances)
            
                     
                df1 = pd.DataFrame(X_testN, columns = features) 
                df2 = pd.DataFrame(y_test, columns = ['cod_lit'])
                df3 = pd.DataFrame(y_pred, columns = ['cod_lit_test'])
                result = pd.concat([df1, df2, df3], axis=1, sort=False)
                namer = config.saveDir+"dataset0/result_RF_predict_"+str(method_inter[u])+"_dataset0.csv"
                result.to_csv(namer)
                
                df22 = pd.read_csv(namer,sep=",")
                df11 = pd.read_csv(name_file,sep=",")
                result = pd.merge(df11, df22, how='left')
                namerr = config.saveDir+"dataset0/join_RF_predict_"+str(method_inter[u])+"_dataset0.csv"
                result.to_csv(namerr)


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
                    plt.xlim(-0.5, 8.5) #figure size according to quantity of cod_lit
                    plt.ylim(8.5, -0.5) #figure size according to quantity of cod_lit

                    
                    
                    plt.tight_layout()
                    plt.ylabel('Predicted label')
                    plt.xlabel('True label')
                    plt.savefig(config.results_graphics + "dataset0/ConfMatrix_dataset0_"+method_inter[u]+".png",dpi=96)
                    
                # Compute confusion matrix

                cnf_matrix = confusion_matrix(y_test, y_pred)
                np.set_printoptions(precision=2)

                class_names = ['0','2','3','5','8','9','11','12','13'] #ajust cod_lit

                plt.figure(figsize=(8, 6))



                plot_confusion_matrix(cnf_matrix, classes=class_names,
                                      title='Confusion matrix')

                #plt.show()