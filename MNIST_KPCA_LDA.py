#=====================================================================================================================
import time
import pickle
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,Normalizer
import warnings
warnings.filterwarnings("ignore")
#=====================================================================================================================
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
#select_data='MNIST_data.csv'
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
# SELECT A SMALL SUBSET(20%) OUT OF 60000 TRAINING SAMPLES THAT EQUAL TO 12000 (1200 from each class) 
# AND (20%) OUT OF 10000 TEST SAMPLES THAT EQUAL TO 2000 (200 from each class) 
 
# create a temporary dataframe that has as alast columns the labels
tmp = pd.concat([x_train, y_train], axis=1)
# name the last column Number
tmp.columns = [*tmp.columns[:-1], 'Class']
# select 100 from each category in the column Number
x_train_small=tmp.groupby('Class').apply(lambda s: s.sample(1200))
# keep only the second index
x_train_small.index = x_train_small.index.get_level_values(1)
# take the last column
y_train_small=x_train_small.iloc[0:x_train_small.shape[0],x_train_small.shape[1]-1:x_train_small.shape[1]]
# drop the last cplumn
x_train_small.drop(['Class'], axis=1)

# create a temporary dataframe that has as alast columns the labels
tmp2 = pd.concat([x_test, y_test], axis=1)
# name the last column Number
tmp2.columns = [*tmp2.columns[:-1], 'Class']
# select 100 from each category in the column Number
x_test_small=tmp2.groupby('Class').apply(lambda s: s.sample(200))
# keep only the second index
x_test_small.index = x_test_small.index.get_level_values(1)
# take the last column
y_test_small=x_test_small.iloc[0:x_test_small.shape[0],x_test_small.shape[1]-1:x_test_small.shape[1]]
# drop the last column
x_test_small=x_test_small.drop(['Class'], axis=1)
x_train_small=x_train_small.drop(['Class'], axis=1)

#=====================================================================================================================
# DIVIDE THE SMALL SUBSET INTO TRAINSET (60%) AND TESTSET (40%), RESULTING IN 7200 TRAIN AND 4800 TEST SAMPLES
test_size_percentage=0.4
x_train, x_test, y_train, y_test = train_test_split(x_train_small, y_train_small,test_size=test_size_percentage, stratify=y_train_small,random_state=1)
#=====================================================================================================================
# Construct pipeline for SCALER-KPCA-LDA-KNN
steps = [('scaler', Normalizer()),('kpca', KernelPCA()), ('lda', LDA()),('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps) # define the pipeline object.
parameters = {'kpca__n_components':[120,250,500],
               'kpca__kernel': ['rbf','poly'],
               'kpca__gamma': [0.5,5,8],
               'lda__n_components':[3,5,9],
               'knn__n_neighbors':[8,10,12]
               }
grid = GridSearchCV(pipeline, param_grid=parameters,scoring='accuracy', cv=3) #1751.5107828 seconds.
grid.fit(x_train, y_train)
# Best params
print('\nBest params:\n', grid.best_params_)
#grid.best_score_
results_df=pd.DataFrame.from_dict(grid.cv_results_)
## save the model to disk
#filename = 'grid_final.sav'
#pickle.dump(grid, open(filename, 'wb'))
## load the model from disk
#grid = pickle.load(open(filename, 'rb'))
#result = grid.score(x_test, y_test)
#print(result)

# Construct pipeline for SCALER-KPCA-LDA-NCC
steps2 = [('scaler', Normalizer()),('kpca', KernelPCA()), ('lda', LDA()),('ncc', NearestCentroid())]
pipeline2 = Pipeline(steps2) # define the pipeline object.
parameters2 = {'kpca__n_components':[120,250,500],
               'kpca__kernel': ['rbf','poly'],
               'kpca__gamma': [5,10,15],
               'lda__n_components':[3,5,9],
               'ncc__metric':['euclidean'],
               'ncc__shrink_threshold':[None, 0.1]
               }
grid2 = GridSearchCV(pipeline2, param_grid=parameters2,scoring='accuracy', cv=3)
grid2.fit(x_train, y_train)
# Best params
print('\nBest params:\n', grid2.best_params_)
results2_df=pd.DataFrame.from_dict(grid2.cv_results_)
## save the model to disk
#filename2 = 'grid2_final.sav'
#pickle.dump(grid2, open(filename2, 'wb'))
## load the model from disk
#grid2 = pickle.load(open(filename2, 'rb'))
#result = grid2.score(x_test, y_test)
#print(result)

# Construct pipeline for SCALER-KPCA-LDA-SVM
steps3 = [('scaler', Normalizer()),('kpca', KernelPCA()), ('lda', LDA()),('svm', svm.SVC(kernel='linear'))]
pipeline3 = Pipeline(steps3) # define the pipeline object.
parameters3 = {'kpca__n_components':[120,250,500],
               'kpca__kernel': ['rbf','poly'],
               'kpca__gamma': [0.5,5,8],
               'lda__n_components':[3,5,9],
               'svm__C':[0.5,5,10],
               'svm__gamma':[0.05,0.5,5]
               }
grid3 = GridSearchCV(pipeline3, param_grid=parameters3,scoring='accuracy', cv=3)
grid3.fit(x_train, y_train)
# Best accuracy
#print('Best training accuracy: %.3f' % grid3.best_score_)
#training_score_kpca_lda_svm=grid3.best_score_
# Best params
print('\nBest params:\n', grid3.best_params_)
results3_df=pd.DataFrame.from_dict(grid3.cv_results_)
## save the model to disk
#filename3 = 'grid3_final.sav'
#pickle.dump(grid3, open(filename3, 'wb'))
## load the model from disk
#grid3 = pickle.load(open(filename3, 'rb'))
#result = grid3.score(x_test, y_test)
#print(result)
#=====================================================================================================================
# SCALE THE DATA DEPENDING ON THEIR DISTRIBUTION 
scaler='normalizer'
select_data='MNIST'
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
print('You have selected tha dataset: '+ select_data +',\ndivided it into a trainset('+str(1-test_size_percentage)+'%) and a testset('+str(test_size_percentage)+'%),\nand scaled the data with the scaler : ' + scaler+'.')
#=====================================================================================================================
# define a dictionary to keep the temporal scores of each model
time_dict_training= {'kpca_lda_knn':'-','kpca_lda_ncc':'-','kpca_lda_svm':'-','svm_without_kpca_lda':'-'}
#=====================================================================================================================
# define new model with optimal components taken from grid.best_params_
kpca = KernelPCA(kernel=grid.best_params_['kpca__kernel'],n_components=grid.best_params_['kpca__n_components'],gamma=grid.best_params_['kpca__gamma'])
# fit and transform x_train_rescaled
start_time_train_kpca_lda_knn = time.perf_counter()
x_train_rescaled_after_kpca = kpca.fit(x_train_rescaled,y_train).transform(x_train_rescaled)
end_time_train_kpca_lda_knn = time.perf_counter()
elapsed_time_train_kpca_lda_knn= end_time_train_kpca_lda_knn - start_time_train_kpca_lda_knn
x_train_rescaled_after_kpca_df=pd.DataFrame(x_train_rescaled_after_kpca,index=x_train.index)
# transform x_test_rescaled
x_test_rescaled_after_kpca=kpca.transform(x_test_rescaled)
x_test_rescaled_after_kpca_df=pd.DataFrame(x_test_rescaled_after_kpca,index=x_test.index)

# creating an LDA object
lda = LDA(n_components=grid.best_params_['lda__n_components'])
# learning the projection matrix
start_time_train_kpca_lda_knn2 = time.perf_counter()
lda=lda.fit(x_train_rescaled_after_kpca, y_train)
# using the model to project x_train_rescaled_after_kpca
x_train_rescaled_after_kpca_and_lda=lda.transform(x_train_rescaled_after_kpca)
# using the model to project x_test_rescaled_after_kpca
end_time_train_kpca_lda_knn2 = time.perf_counter()
elapsed_time_train_kpca_lda_knn2= end_time_train_kpca_lda_knn2 - start_time_train_kpca_lda_knn2
x_test_rescaled_after_kpca_and_lda = lda.transform(x_test_rescaled_after_kpca)
# make them dataframes
x_train_rescaled_after_kpca_and_lda_df=pd.DataFrame(x_train_rescaled_after_kpca_and_lda,index=x_train.index)
x_test_rescaled_after_kpca_and_lda_df=pd.DataFrame(x_test_rescaled_after_kpca_and_lda,index=x_test.index)

knn = KNeighborsClassifier(n_neighbors=grid.best_params_['knn__n_neighbors'])
start_time_knn = time.perf_counter()
knn.fit(x_train_rescaled_after_kpca_and_lda,y_train)
end_time_knn = time.perf_counter()
elapsed_time_train_kpca_lda_knn3= end_time_knn - start_time_knn
# update the time dictionary for training the model : kpca_lda_knn
time_dict_training.update(kpca_lda_knn = elapsed_time_train_kpca_lda_knn+elapsed_time_train_kpca_lda_knn2+elapsed_time_train_kpca_lda_knn3)
y_predicted_knn = knn.predict(x_test_rescaled_after_kpca_and_lda)
y_predicted_knn=pd.DataFrame(y_predicted_knn,index=x_test_rescaled.index)
training_score_kpca_lda_knn=knn.score(x_train_rescaled_after_kpca_and_lda, y_train)
testing_score_kpca_lda_knn=knn.score(x_test_rescaled_after_kpca_and_lda, y_test)
#=====================================================================================================================
# define new model with optimal components taken from grid2.best_params_
kpca2 = KernelPCA(kernel=grid2.best_params_['kpca__kernel'],n_components=grid2.best_params_['kpca__n_components'],gamma=grid2.best_params_['kpca__gamma'])
# fit and transform x_train_rescaled
start_time_train_kpca_lda_ncc = time.perf_counter()
x_train_rescaled_after_kpca2 = kpca2.fit(x_train_rescaled,y_train).transform(x_train_rescaled)
end_time_train_kpca_lda_ncc = time.perf_counter()
elapsed_time_train_kpca_lda_ncc= end_time_train_kpca_lda_ncc - start_time_train_kpca_lda_ncc
x_train_rescaled_after_kpca2_df=pd.DataFrame(x_train_rescaled_after_kpca2,index=x_train.index)
# transform x_test_rescaled
x_test_rescaled_after_kpca2=kpca2.transform(x_test_rescaled)
x_test_rescaled_after_kpca2_df=pd.DataFrame(x_test_rescaled_after_kpca2,index=x_test.index)

# creating an LDA object
lda2 = LDA(n_components=grid2.best_params_['lda__n_components'])
# learning the projection matrix
start_time_train_kpca_lda_ncc2 = time.perf_counter()
lda2=lda2.fit(x_train_rescaled_after_kpca2, y_train)
# using the model to project x_train_rescaled_after_kpca
x_train_rescaled_after_kpca_and_lda2=lda2.transform(x_train_rescaled_after_kpca2)
# using the model to project x_test_rescaled_after_kpca
end_time_train_kpca_lda_ncc2 = time.perf_counter()
elapsed_time_train_kpca_lda_ncc2= end_time_train_kpca_lda_ncc2 - start_time_train_kpca_lda_ncc2
x_test_rescaled_after_kpca_and_lda2 = lda2.transform(x_test_rescaled_after_kpca2)
# make them dataframes
x_train_rescaled_after_kpca_and_lda2_df=pd.DataFrame(x_train_rescaled_after_kpca_and_lda2,index=x_train.index)
x_test_rescaled_after_kpca_and_lda2_df=pd.DataFrame(x_test_rescaled_after_kpca_and_lda2,index=x_test.index)

ncc = NearestCentroid(metric=grid2.best_params_['ncc__metric'],shrink_threshold=grid2.best_params_['ncc__shrink_threshold'])
start_time_ncc = time.perf_counter()
ncc.fit(x_train_rescaled_after_kpca_and_lda2,y_train)
end_time_ncc = time.perf_counter()
elapsed_time_train_kpca_lda_ncc3= end_time_ncc - start_time_ncc
# update the time dictionary for training the model : kpca_lda_knn
time_dict_training.update(kpca_lda_ncc = elapsed_time_train_kpca_lda_ncc+elapsed_time_train_kpca_lda_ncc2+elapsed_time_train_kpca_lda_ncc3)
y_predicted_ncc = ncc.predict(x_test_rescaled_after_kpca_and_lda2)
y_predicted_ncc=pd.DataFrame(y_predicted_ncc,index=x_test_rescaled.index)
training_score_kpca_lda_ncc=ncc.score(x_train_rescaled_after_kpca_and_lda2, y_train)
testing_score_kpca_lda_ncc=ncc.score(x_test_rescaled_after_kpca_and_lda2, y_test)
#=====================================================================================================================
# define new model with optimal components taken from grid3.best_params_
kpca3 = KernelPCA(kernel=grid3.best_params_['kpca__kernel'],n_components=grid3.best_params_['kpca__n_components'],gamma=grid3.best_params_['kpca__gamma'])
# fit and transform x_train_rescaled
start_time_train_kpca_lda_svm = time.perf_counter()
x_train_rescaled_after_kpca3 = kpca3.fit(x_train_rescaled,y_train).transform(x_train_rescaled)
end_time_train_kpca_lda_svm = time.perf_counter()
elapsed_time_train_kpca_lda_svm= end_time_train_kpca_lda_svm - start_time_train_kpca_lda_svm
x_train_rescaled_after_kpca3_df=pd.DataFrame(x_train_rescaled_after_kpca3,index=x_train.index)
# transform x_test_rescaled
x_test_rescaled_after_kpca3=kpca3.transform(x_test_rescaled)
x_test_rescaled_after_kpca3_df=pd.DataFrame(x_test_rescaled_after_kpca3,index=x_test.index)

# creating an LDA object
lda3 = LDA(n_components=grid3.best_params_['lda__n_components'])
# learning the projection matrix
start_time_train_kpca_lda_svm2 = time.perf_counter()
lda3=lda3.fit(x_train_rescaled_after_kpca3, y_train)
# using the model to project x_train_rescaled_after_kpca
x_train_rescaled_after_kpca_and_lda3=lda3.transform(x_train_rescaled_after_kpca3)
# using the model to project x_test_rescaled_after_kpca
end_time_train_kpca_lda_svm2 = time.perf_counter()
elapsed_time_train_kpca_lda_svm2= end_time_train_kpca_lda_svm2 - start_time_train_kpca_lda_svm2
x_test_rescaled_after_kpca_and_lda3 = lda3.transform(x_test_rescaled_after_kpca3)
# make them dataframes
x_train_rescaled_after_kpca_and_lda3_df=pd.DataFrame(x_train_rescaled_after_kpca_and_lda3,index=x_train.index)
x_test_rescaled_after_kpca_and_lda3_df=pd.DataFrame(x_test_rescaled_after_kpca_and_lda3,index=x_test.index)

svm =SVC(kernel='linear', C=grid3.best_params_['svm__C'],gamma=grid3.best_params_['svm__gamma'])
start_time_svm = time.perf_counter()
svm.fit(x_train_rescaled_after_kpca_and_lda3,y_train)
end_time_svm = time.perf_counter()
elapsed_time_train_kpca_lda_svm3= end_time_svm - start_time_svm
# update the time dictionary for training the model : kpca_lda_knn
time_dict_training.update(kpca_lda_svm = elapsed_time_train_kpca_lda_svm+elapsed_time_train_kpca_lda_svm2+elapsed_time_train_kpca_lda_svm3)
y_predicted_svm = svm.predict(x_test_rescaled_after_kpca_and_lda3)
y_predicted_svm=pd.DataFrame(y_predicted_svm,index=x_test_rescaled.index)
training_score_kpca_lda_svm=svm.score(x_train_rescaled_after_kpca_and_lda3, y_train)
testing_score_kpca_lda_svm=svm.score(x_test_rescaled_after_kpca_and_lda3, y_test)
#=====================================================================================================================
# compare with the best svm model created in the first project
svm_without_kpca_lda = SVC(kernel='rbf',C=5,gamma=5)
start_time_train_svm_without_kpca_lda = time.perf_counter()
svm_without_kpca_lda.fit(x_train_rescaled,y_train)
end_time_train_svm_without_kpca_lda = time.perf_counter()
elapsed_time_train_svm_without_kpca_lda= end_time_train_svm_without_kpca_lda - start_time_train_svm_without_kpca_lda

time_dict_training.update(svm_without_kpca_lda = elapsed_time_train_svm_without_kpca_lda)
y_predicted_svm_without_kpca_lda = svm_without_kpca_lda.predict(x_test_rescaled)
y_predicted_svm_without_kpca_lda=pd.DataFrame(y_predicted_svm_without_kpca_lda,index=x_test_rescaled.index)

training_score_svm_without_kpca_lda=svm_without_kpca_lda.score(x_train_rescaled, y_train)
testing_score_svm_without_kpca_lda=svm_without_kpca_lda.score(x_test_rescaled, y_test)
#=====================================================================================================================
# SORT SPEED DICTIONARY BETWEEN KPCA-LDA-KNN , KPCA-LDA-NCC , KPCA-LDA-LINEAR SVM and SVM
time_dict_training_sorted=sorted(time_dict_training.items(),key=operator.itemgetter(1))
print('Speed sorted by order after training KPCA-LDA-KNN , KPCA-LDA-NCC , KPCA-LDA-LINEAR SVM and SVM algorithms.')
print('1. Fastest        : '+ str(time_dict_training_sorted[0]) +'.\n2. Of medium speed: '+ str(time_dict_training_sorted[1])+'.\n3. Of medium speed: '+ str(time_dict_training_sorted[2])+'.\n4. Slowest        : '+ str(time_dict_training_sorted[3])+'.')
#=====================================================================================================================
# SORT ACCURACY DICTIONARY TRAINING BETWEEN KPCA-LDA-KNN , KPCA-LDA-NCC , KPCA-LDA-LINEAR SVM and SVM
accuracy_dict_training= {'kpca_lda_knn':'-','kpca_lda_ncc':'-','kpca_lda_svm':'-','svm_without_kpca_lda':'-'}
accuracy_dict_training.update(kpca_lda_knn=training_score_kpca_lda_knn)
accuracy_dict_training.update(kpca_lda_ncc=training_score_kpca_lda_ncc)
accuracy_dict_training.update(kpca_lda_svm=training_score_kpca_lda_svm)
accuracy_dict_training.update(svm_without_kpca_lda=training_score_svm_without_kpca_lda)
accuracy_training_sorted=sorted(accuracy_dict_training.items(),key=operator.itemgetter(1))
print('Accuracy sorted by order after training KPCA+LDA,SVM,KNN and NCC algorithms.')
print('1. Most accurate(training): '+ str(accuracy_training_sorted[3]) +'.\n2. Of medium accuracy(training): '+ str(accuracy_training_sorted[2])+'.\n3. Of medium accuracy(training): '+ str(accuracy_training_sorted[1])+'.\n4. Least accurate(training): '+ str(accuracy_training_sorted[0])+'.')
#=====================================================================================================================
# SORT ACCURACY DICTIONARY TESTING BETWEEN KPCA-LDA-KNN , KPCA-LDA-NCC , KPCA-LDA-LINEAR SVM and SVM
accuracy_dict_testing= {'kpca_lda_knn':'-','kpca_lda_ncc':'-','kpca_lda_svm':'-','svm_without_kpca_lda':'-'}
accuracy_dict_testing.update(kpca_lda_knn=testing_score_kpca_lda_knn)
accuracy_dict_testing.update(kpca_lda_ncc=testing_score_kpca_lda_ncc)
accuracy_dict_testing.update(kpca_lda_svm=testing_score_kpca_lda_svm)
accuracy_dict_testing.update(svm_without_kpca_lda=testing_score_svm_without_kpca_lda)
accuracy_testing_sorted=sorted(accuracy_dict_testing.items(),key=operator.itemgetter(1))
print('Accuracy sorted by order after testing KPCA+LDA,SVM,KNN and NCC algorithms.')
print('1. Most accurate(testing): '+ str(accuracy_testing_sorted[3]) +'.\n2. Of medium accuracy(testing): '+ str(accuracy_testing_sorted[2])+'.\n3. Of medium accuracy(testing): '+ str(accuracy_testing_sorted[1])+'.\n4. Least accurate(testing): '+ str(accuracy_testing_sorted[0])+'.')
#=====================================================================================================================
y_predicted_svm.columns=['Class']
y_predicted_knn.columns=['Class']
y_predicted_ncc.columns=['Class']
y_predicted_svm_without_kpca_lda.columns=['Class']
def find_if_wrong_class(y_predicted,y_test):
    for i in list(y_predicted.index):
        pred = y_predicted['Class'][i]
        true= y_test['Class'][i]
        if pred != true:
            print("The predicted label is: %s, but the true label is: %s, located in index: %d" % (pred, true,i))
            break
        else:
            print("Correct classification!")
find_if_wrong_class(y_predicted_svm,y_test)
find_if_wrong_class(y_predicted_knn,y_test)
find_if_wrong_class(y_predicted_ncc,y_test)
find_if_wrong_class(y_predicted_svm_without_kpca_lda,y_test)
#=====================================================================================================================
# PLOT THE CONFUSION MATRIX
def computeAndPlotCM(y_test,y_predicted):
    cm=confusion_matrix(y_test,y_predicted)
    def get_df_name(df):
        name =[x for x in globals() if globals()[x] is df][0]
        return name
    name=(get_df_name(y_predicted))
    class_names=['0','1','2','3','4','5','6','7','8','9']
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
    return cm
cm_svm=computeAndPlotCM(y_test,y_predicted_svm)
cm_knn=computeAndPlotCM(y_test,y_predicted_knn)
cm_ncc=computeAndPlotCM(y_test,y_predicted_ncc)
cm_kpca_lda=computeAndPlotCM(y_test,y_predicted_svm_without_kpca_lda)
#=====================================================================================================================
# FIND HOW MUCH TIME HAS ELAPSED
end_mnist = time.perf_counter()
elapsed_mnist = end_mnist - start_mnist
print('The program finished in : '+ str(elapsed_mnist)+' seconds.') #25314.062644300007 seconds. ~7 hours
#=====================================================================================================================