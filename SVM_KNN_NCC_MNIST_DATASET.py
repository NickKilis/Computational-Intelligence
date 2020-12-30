#=====================================================================================================================
import time
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,Normalizer
# START THE OVERALL TIMER
start_mnist = time.perf_counter()
#=====================================================================================================================
# LOAD DATA WITH TENSORFLOW (REQUIRES INSTALLATION FIRST)
#import tensorflow as tf
#from mnist import MNIST
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0
#=====================================================================================================================
# LOAD DATA (from: https://data.world/nrippner/mnist-handwritten-digits)
select_data='MNIST_data.csv'
#data=pd.read_csv(select_data)
#select_target_data='MNIST_target.csv'
#target=pd.read_csv(select_target_data)
select_train_data='MNIST_data_train.csv'
select_test_data='MNIST_data_test.csv'
select_train_target='MNIST_target_train.csv'
select_test_target='MNIST_target_test.csv'
x_train=pd.read_csv(select_train_data)
x_test=pd.read_csv(select_test_data)
y_train=pd.read_csv(select_train_target)
y_test=pd.read_csv(select_test_target)
#=====================================================================================================================
# SCALE THE DATA DEPENDING ON THEIR DISTRIBUTION 
# PCA is effected by scale so we need to scale the features in your data before applying PCA.
test_size_percentage=(len(x_test)/len(x_train))*100
scaler='normalizer'
if scaler=='standard':
    standard = StandardScaler()
    standard.fit(x_train)
    x_train_rescaled =standard.transform(x_train)
    x_train_rescaled=pd.DataFrame(x_train_rescaled,index=x_train.index)
    x_test_rescaled=standard.transform(x_test)
    x_test_rescaled=pd.DataFrame(x_test_rescaled,index=x_test.index)
elif scaler=='minmax':
    minmax = MinMaxScaler()
    minmax.fit(x_train)
    x_train_rescaled = minmax.transform(x_train)
    x_train_rescaled=pd.DataFrame(x_train_rescaled,index=x_train.index)
    x_test_rescaled=minmax.transform(x_test)
    x_test_rescaled=pd.DataFrame(x_test_rescaled,index=x_test.index)
elif scaler=='robust':
    robust = RobustScaler()
    robust.fit(x_train)
    x_train_rescaled = robust.transform(x_train)
    x_train_rescaled=pd.DataFrame(x_train_rescaled,index=x_train.index)
    x_test_rescaled=robust.transform(x_test)
    x_test_rescaled=pd.DataFrame(x_test_rescaled,index=x_test.index)
elif scaler=='normalizer':
    normalizer = Normalizer()
    normalizer.fit(x_train)
    x_train_rescaled = normalizer.transform(x_train)
    x_train_rescaled=pd.DataFrame(x_train_rescaled,index=x_train.index)
    x_test_rescaled=normalizer.transform(x_test)
    x_test_rescaled=pd.DataFrame(x_test_rescaled,index=x_test.index)
else:
    print('''You have selected an invalid scaler!
    The valid scaler names are:
    standard  : xi-mean(x)/stdev(x)     (For normally distributed data.)
    minmax    : xi-min(x)/max(x)-min(x) (For non-Gaussian or with very small standard deviation and without outliers.)
    robust    : xi-Q1(x)/Q3(x)-Q1(x)    (More suitable for when there are outliers in the data.)
    normalizer: xi/sqrt(xi^2+yi^2+zi^2) (All the points are brought within a sphere with radius 1.)
    ''')
print('You have selected tha dataset: '+ select_data +',\ndivided it into a trainset('+str(100-test_size_percentage)+'%) and a testset('+str(test_size_percentage)+'%),\nand scaled the data with the scaler : ' + scaler+'.')
#=====================================================================================================================
## see an example before normalization
#o=x_train.loc[0]
#fig =plt.figure(figsize = (8,8))
#plt.imshow(o.values.reshape(28,28),cmap=plt.cm.binary)
#plt.show()
## see an example after normalization
##if oo is an array and not a dataframe,so we dont use "loc" or "values"
#oo=x_train_rescaled.loc[0]
#fig =plt.figure(figsize = (8,8))
#plt.imshow(oo.values.reshape(28,28),cmap=plt.cm.binary)
#plt.show()
#=====================================================================================================================
components=100
pca = PCA(n_components=components)
x_train_rescaled_after_pca = pca.fit_transform(x_train_rescaled)
x_test_rescaled_after_pca = pca.transform(x_test_rescaled)

#print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
fig6 = plt.figure(figsize = (8,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.show()
#Together, the two components contain (total_percentage)% of the information.
total_percentage=0.0
for i in range(0,components):
    total_percentage = total_percentage+ pca.explained_variance_ratio_[i]
print('Together, the '+str(components)+ ' components contain: ('+ str(total_percentage) +'%) of the initial information.')
#=====================================================================================================================
scores = ['accuracy']
grid_is_good=False
if grid_is_good == True: 
    for score in scores:
        print("Performing Grid search for %s...\n" % score)
        # GRID SEARCH
        C_range=np.array([0.5,5,10])
        gamma_range = np.array([0.05,0.5,5])
        parameters_rbf = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}
        model_rbf = svm.SVC()
        # increase n_jobs in order to run in parallel
        grid_clsf_rbf = GridSearchCV(estimator=model_rbf,param_grid=parameters_rbf,cv=3, verbose=2)
        start_time = time.perf_counter()
        print('Start param searching (rbf) at {}'.format(str(start_time)))
        grid_clsf_rbf.fit(x_train_rescaled_after_pca, y_train.values.ravel())
        elapsed_time= time.perf_counter() - start_time  # 2155.198885679245 seconds.
        print('Elapsed training time (rbf), param searching {}'.format(str(elapsed_time)))
        sorted(grid_clsf_rbf.cv_results_.keys())
        classifier_rbf = grid_clsf_rbf.best_estimator_
        params_rbf = grid_clsf_rbf.best_params_
        scores_rbf = grid_clsf_rbf.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))
        
        print("Grid scores on development set:")
        means = grid_clsf_rbf.cv_results_['mean_test_score']
        stds = grid_clsf_rbf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_clsf_rbf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        y_true, y_pred = y_test, grid_clsf_rbf.predict(x_test_rescaled_after_pca)
        print("Classification report for classifier %s:\n%s\n"% (grid_clsf_rbf,classification_report(y_true, y_pred)))
        print("Accuracy={}".format(accuracy_score(y_true, y_pred)))
else:
    pass
grid_is_good2=False
if grid_is_good2 == True:
    for score in scores:
        print("Performing Grid search for %s...\n" % score)
        # GRID SEARCH
        C_range=np.array([0.5,5,10])
        parameters_linear = {'kernel':['linear'], 'C':C_range}
        model_linear = svm.SVC()
        # increase n_jobs in order to run in parallel
        grid_clsf_linear = GridSearchCV(estimator=model_linear,param_grid=parameters_linear,cv=3, verbose=2)
        start_time = time.perf_counter()
        print('Start param searching (linear) at {}'.format(str(start_time)))
        grid_clsf_linear.fit(x_train_rescaled_after_pca, y_train.values.ravel())
        elapsed_time= time.perf_counter() - start_time #334.9253661632538 seconds.
        print('Elapsed training time (linear), param searching {}'.format(str(elapsed_time)))
        sorted(grid_clsf_linear.cv_results_.keys())
        classifier_linear = grid_clsf_linear.best_estimator_
        params_linear = grid_clsf_linear.best_params_
        scores_linear = grid_clsf_linear.cv_results_['mean_test_score'].reshape(len(C_range))        
        
        print("Grid scores on development set:")
        means = grid_clsf_linear.cv_results_['mean_test_score']
        stds = grid_clsf_linear.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_clsf_linear.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        y_true, y_pred = y_test, grid_clsf_linear.predict(x_test_rescaled_after_pca)
        print("Classification report for classifier %s:\n%s\n"% (grid_clsf_linear,classification_report(y_true, y_pred)))
        print("Accuracy={}".format(accuracy_score(y_true, y_pred)))
else:
    pass
grid_is_good3=False
if grid_is_good3 == True:
    for score in scores:
        print("Performing Grid search for %s...\n" % score)
        # GRID SEARCH
        C_range=np.array([0.5,5,10])
        gamma_range = np.array([0.05,0.5,5])
        parameters_sigmoid = {'kernel':['sigmoid'], 'C':C_range, 'gamma': gamma_range}
        model_sigmoid = svm.SVC()
        # increase n_jobs in order to run in parallel
        grid_clsf_sigmoid = GridSearchCV(estimator=model_sigmoid,param_grid=parameters_sigmoid,cv=3, verbose=2)
        start_time = time.perf_counter()
        print('Start param searching (sigmoid) at {}'.format(str(start_time)))
        grid_clsf_sigmoid.fit(x_train_rescaled_after_pca, y_train.values.ravel())
        elapsed_time= time.perf_counter()- start_time # 2122.203788280487 seconds.
        print('Elapsed training time (sigmoid), param searching {}'.format(str(elapsed_time)))
        sorted(grid_clsf_sigmoid.cv_results_.keys())
        classifier_sigmoid = grid_clsf_sigmoid.best_estimator_
        params_sigmoid = grid_clsf_sigmoid.best_params_
        scores_sigmoid = grid_clsf_sigmoid.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))        
        
        print("Grid scores on development set:")
        means = grid_clsf_sigmoid.cv_results_['mean_test_score']
        stds = grid_clsf_sigmoid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_clsf_sigmoid.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        y_true, y_pred = y_test, grid_clsf_sigmoid.predict(x_test_rescaled_after_pca)
        print("Classification report for classifier %s:\n%s\n"% (grid_clsf_sigmoid,classification_report(y_true, y_pred)))
        print("Accuracy={}".format(accuracy_score(y_true, y_pred)))
else:
    pass
if grid_is_good & grid_is_good2 & grid_is_good3==False:
      model = SVC(kernel='rbf',C=5,gamma=5)
else:
#=====================================================================================================================
    # DEFINE THE BEST SVM MODEL
    model_dict_scores= {'rbf':'-','linear':'-','sigmoid':'-'}
    model_dict_scores.update(rbf=np.amax(scores_rbf))
    model_dict_scores.update(linear=np.amax(scores_linear))
    model_dict_scores.update(sigmoid=np.amax(scores_sigmoid))    
    model_dict_scores_sorted=sorted(model_dict_scores.items(),key=operator.itemgetter(1))
    print('Accuracy sorted by order after training SVM,KNN and NCC algorithms.')
    print('1. Most accurate(training): '+ str(model_dict_scores_sorted[2]) +'.\n2. Of medium accuracy(training): '+ str(model_dict_scores_sorted[1])+'.\n3. Least accurate(training): '+ str(model_dict_scores_sorted[0])+'.')
    (best_model,best_accuracy)=model_dict_scores_sorted[2]
    
    def get_key(val,dictionary):
        best_parameters={}
        for key,value in dictionary.items(): 
             if val == value: 
                 best_parameters=dictionary
                 return best_parameters
        return "key doesn't exist"
    best_parameters1=get_key(best_model,params_linear)
    best_parameters2=get_key(best_model,params_sigmoid)
    best_parameters3=get_key(best_model,params_rbf)
    best_parameters={}
    if type(best_parameters1) == type(dict()):
        best_parameters=best_parameters1
    elif type(best_parameters2) == type(dict()):
        best_parameters=best_parameters2
    elif type(best_parameters3) == type(dict()):
        best_parameters=best_parameters3
    else:
        print('Something went wrong...all of them were strings!')
    
    if 'gamma' in best_parameters:
      model = SVC(kernel=best_model,C=best_parameters['C'],gamma=best_parameters['gamma'])
    else:
      model = SVC(kernel=best_model,C=best_parameters['C'])
#=====================================================================================================================
# define a dictionary to keep the temporal scores of each model
time_dict_training= {'svm':'-','knn':'-','ncc':'-'} 
# TRAIN THE BEST SVM MODEL
start_time_train = time.perf_counter()
print('Start training at {}'.format(str(start_time_train)))
model.fit(x_train_rescaled_after_pca,y_train.values.ravel())
end_time_train = time.perf_counter()
print( 'Stop training {}'.format(str(end_time_train)))
elapsed_time_train= end_time_train - start_time_train
print('Elapsed training {}'.format(str(elapsed_time_train)))
time_dict_training.update(svm = elapsed_time_train)
# PREDICT WITH THE BEST MODEL
start_time_predict = time.perf_counter()
print('Start predicting at {}'.format(str(start_time_predict)))
y_predicted_svm = model.predict(x_test_rescaled_after_pca)
end_time_predict = time.perf_counter()
print( 'Stop predicting {}'.format(str(end_time_predict)))
elapsed_time_predict= end_time_predict - start_time_predict
print('Elapsed predicting {}'.format(str(elapsed_time_predict)))
# PRINT THE PREDICTED CLASSES
training_score_svm=model.score(x_train_rescaled_after_pca, y_train)
testing_score_svm=model.score(x_test_rescaled_after_pca, y_test)
#=====================================================================================================================
# K-NEAREST NEIGHBOUR
knn = KNeighborsClassifier(n_neighbors=5)
start_time_knn = time.perf_counter()
print('Start knn training at {}'.format(str(start_time_knn)))
knn.fit(x_train_rescaled_after_pca,y_train.values.ravel())
end_time_knn = time.perf_counter()
elapsed_time_train_knn= end_time_knn - start_time_knn
print('Elapsed knn training {}'.format(str(elapsed_time_train_knn)))
time_dict_training.update(knn=elapsed_time_train_knn)
start_time_predict_knn = time.perf_counter()
y_predicted_knn = knn.predict(x_test_rescaled_after_pca)
end_time_predict_knn = time.perf_counter()
elapsed_time_predict_knn= end_time_predict_knn - start_time_predict_knn
training_score_knn=knn.score(x_train_rescaled_after_pca, y_train)
testing_score_knn=knn.score(x_test_rescaled_after_pca, y_test)
#=====================================================================================================================
# NEAREST CENTROID
ncc = NearestCentroid(metric='euclidean', shrink_threshold=None)
start_time_ncc = time.perf_counter()
print('Start ncc training at {}'.format(str(start_time_ncc)))
ncc.fit(x_train_rescaled_after_pca,y_train.values.ravel())
end_time_ncc = time.perf_counter()
elapsed_time_train_ncc= end_time_ncc - start_time_ncc
print('Elapsed ncc training {}'.format(str(elapsed_time_train_ncc)))
time_dict_training.update(ncc=elapsed_time_train_ncc)
start_time_predict_ncc = time.perf_counter()
y_predicted_ncc = ncc.predict(x_test_rescaled_after_pca)
end_time_predict_ncc = time.perf_counter()
elapsed_time_predict_ncc= end_time_predict_ncc - start_time_predict_ncc
training_score_ncc=knn.score(x_train_rescaled_after_pca, y_train)
testing_score_ncc=knn.score(x_test_rescaled_after_pca, y_test)
#=====================================================================================================================
# SORT SPEED DICTIONARY BETWEEN SVM KNN AND NCC
time_dict_training_sorted=sorted(time_dict_training.items(),key=operator.itemgetter(1))
print('Speed sorted by order after training SVM,KNN and NCC algorithms.')
print('1. Fastest        : '+ str(time_dict_training_sorted[0]) +'.\n2. Of medium speed: '+ str(time_dict_training_sorted[1])+'.\n3. Slowest        : \
'+ str(time_dict_training_sorted[2])+'.')
#=====================================================================================================================
# SORT ACCURACY DICTIONARY TRAINING BETWEEN SVM KNN AND NCC
accuracy_dict_training= {'svm':'-','knn':'-','ncc':'-'} 
accuracy_dict_training.update(svm=training_score_svm)
accuracy_dict_training.update(knn=training_score_knn)
accuracy_dict_training.update(ncc=training_score_ncc)
accuracy_training_sorted=sorted(accuracy_dict_training.items(),key=operator.itemgetter(1))
print('Accuracy sorted by order after training SVM,KNN and NCC algorithms.')
print('1. Most accurate(training): '+ str(accuracy_training_sorted[2]) +'.\n2. Of medium accuracy(training): '+ str(accuracy_training_sorted[1])+'.\n3. \
Least accurate(training): '+ str(accuracy_training_sorted[0])+'.')
#=====================================================================================================================
# SORT ACCURACY DICTIONARY TESTING BETWEEN SVM KNN AND NCC
accuracy_dict_testing= {'svm':'-','knn':'-','ncc':'-'} 
accuracy_dict_testing.update(svm=testing_score_svm)
accuracy_dict_testing.update(knn=testing_score_knn)
accuracy_dict_testing.update(ncc=testing_score_ncc)
accuracy_testing_sorted=sorted(accuracy_dict_testing.items(),key=operator.itemgetter(1))
print('Accuracy sorted by order after testing SVM,KNN and NCC algorithms.')
print('1. Most accurate(testing):      '+ str(accuracy_training_sorted[2]) +'.\n2. Of medium accuracy(testing): '+ str(accuracy_testing_sorted[1])+'.\n3. \
Least accurate(testing):     '+ str(accuracy_testing_sorted[0])+'.')
#=====================================================================================================================
# BINARY CLASSIFICATION
## make binary task by doing odd vs even numers
y_predicted_svm=pd.DataFrame(y_predicted_svm)
y_predicted_knn=pd.DataFrame(y_predicted_knn)
y_predicted_ncc=pd.DataFrame(y_predicted_ncc)
y_predicted_svm.columns=['Species']
y_predicted_knn.columns=['Species']
y_predicted_ncc.columns=['Species']
y_test.columns=['Species']

def find_if_wrong_class(y_predicted,y_test,x_test_rescaled_after_pca):
    for i in range(0,len(y_predicted)):
        pred = y_predicted['Species'][i]
        true= y_test['Species'][i]
        if pred != true:
            print("The predicted label is: %s, but the true label is: %s, located in index: %d" % (pred, true,i))
            fig =plt.figure(figsize = (8,8))
            dataFrame=pd.DataFrame(x_test_rescaled_after_pca)
            ooo=dataFrame.loc[i]
            plt.imshow(ooo.values.reshape(10,10),cmap=plt.cm.binary)
            plt.title('Predicted image')
            plt.show()
            fig =plt.figure(figsize = (8,8))
            plt.title('Actual image')
            oooo=x_test.loc[i]
            plt.imshow(oooo.values.reshape(28,28),cmap=plt.cm.binary)
            plt.show()
            break
        else:
            print("Correct classification!")
### UNCOMMENT ONLY FOR SEEING THE CORRECT CLASSIFIED EXAMPLE            
#            fig =plt.figure(figsize = (8,8))
#            dataFrame=pd.DataFrame(x_test_rescaled_after_pca)
#            ooo=dataFrame.loc[i]
#            plt.imshow(ooo.values.reshape(10,10),cmap=plt.cm.binary)
#            plt.title('Predicted image')
#            plt.show()
#            fig =plt.figure(figsize = (8,8))
#            plt.title('Actual image')
#            oooo=x_test.loc[i]
#            plt.imshow(oooo.values.reshape(28,28),cmap=plt.cm.binary)
#            plt.show()
#            break

# non-binary classification
#find_if_wrong_class(y_predicted_svm,y_test,x_test_rescaled_after_pca)
#find_if_wrong_class(y_predicted_knn,y_test,x_test_rescaled_after_pca)
#find_if_wrong_class(y_predicted_ncc,y_test,x_test_rescaled_after_pca)

def predict2binary(y_pred):
    y_pred =y_pred % 2
    y_pred =2*y_pred -1
    new_y=pd.DataFrame(y_pred)  
    return(new_y)

y_predicted_svm_binary=predict2binary(y_predicted_svm)    
y_predicted_knn_binary=predict2binary(y_predicted_knn) 
y_predicted_ncc_binary=predict2binary(y_predicted_ncc)    
y_test_binary=predict2binary(y_test)
# binary classification
find_if_wrong_class(y_predicted_svm_binary,y_test_binary,x_test_rescaled_after_pca)
find_if_wrong_class(y_predicted_knn_binary,y_test_binary,x_test_rescaled_after_pca)
find_if_wrong_class(y_predicted_ncc_binary,y_test_binary,x_test_rescaled_after_pca)
#=====================================================================================================================
# PLOT THE CONFUSION MATRIX
def computeAndPlotCM(y_test,y_predicted):
    cm=confusion_matrix(y_test,y_predicted)
    def get_df_name(df):
        name =[x for x in globals() if globals()[x] is df][0]
        return name
    name=(get_df_name(y_predicted))
    class_names=['Even','Odd']
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names )
    fig = plt.figure(figsize = (8,8))
    plt.title('Confusion Matrix for ' + name+'.') 
    Ηeatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()    

computeAndPlotCM(y_test_binary,y_predicted_svm_binary)
computeAndPlotCM(y_test_binary,y_predicted_knn_binary)
computeAndPlotCM(y_test_binary,y_predicted_ncc_binary)
#=====================================================================================================================
# FIND HOW MUCH TIME HAS ELAPSED
end_mnist = time.perf_counter()
elapsed_mnist = end_mnist - start_mnist
print('The program finished in : '+ str(elapsed_mnist)+' seconds.') # 1324.7618969 seconds.