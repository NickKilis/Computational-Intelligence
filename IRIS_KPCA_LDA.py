import time
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,Normalizer
import warnings
warnings.filterwarnings("ignore")
# START THE OVERALL TIMER
start_iris = time.perf_counter()
##=====================================================================================================================
# LOAD DATA
select_data='Iris.csv'
data=pd.read_csv(select_data)
# DISTINGUISH BETWEEN FEATURES AND TARGETS
rows = data.shape[0]
cols = data.shape[1]
x = data.iloc[0:rows , 0:cols-1]
y = data.iloc[0:rows, cols-1:cols]
# USE LABEL ENCODER
X = x[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y = y['Species'].values
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1
label_dict = {1: 'Iris-Setosa', 2: 'Iris-Versicolor', 3:'Iris-Virginica'}
# VISUALIZE THE DISTRIBUTIONS
#sns.pairplot(data=data,kind='scatter', hue='Species')
# DIVIDE DATA INTO TRAINSET (60%) AND TESTSET (40%)
test_size_percentage=0.4
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=test_size_percentage, stratify=y,random_state=1)
#=====================================================================================================================
# Construct pipeline for SCALER-KPCA-LDA-KNN
steps = [('scaler', StandardScaler()),('kpca', KernelPCA()), ('lda', LDA()),('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps) # define the pipeline object.
parameters = {'kpca__n_components':[1,2,3,4],
               'kpca__kernel': ['linear', 'rbf','poly'],
               'kpca__gamma': [0.01,0.1,0.5,2,5],
               'lda__n_components':[1,2,3],
               'knn__n_neighbors':[3,5,7,10]
               }
grid = GridSearchCV(pipeline, param_grid=parameters,scoring='accuracy', cv=5)
grid.fit(x_train, y_train)
# Best params
print('\nBest params:\n', grid.best_params_)

# Construct pipeline for SCALER-KPCA-LDA-NCC
steps2 = [('scaler', StandardScaler()),('kpca', KernelPCA()), ('lda', LDA()),('ncc', NearestCentroid())]
pipeline2 = Pipeline(steps2) # define the pipeline object.
parameters2 = {'kpca__n_components':[1,2,3,4],
               'kpca__kernel': ['linear', 'rbf','poly'],
               'kpca__gamma': [0.01,0.1,0.5,2,5],
               'lda__n_components':[1,2,3],
               'ncc__metric':['euclidean', 'cosine'],
               'ncc__shrink_threshold':[None, 0.1, 0.5]
               }
grid2 = GridSearchCV(pipeline2, param_grid=parameters2,scoring='accuracy', cv=5)
grid2.fit(x_train, y_train)
# Best params
print('\nBest params:\n', grid2.best_params_)

# Construct pipeline for SCALER-KPCA-LDA-SVM
steps3 = [('scaler', StandardScaler()),('kpca', KernelPCA()), ('lda', LDA()),('svm', svm.SVC(kernel='linear'))]
pipeline3 = Pipeline(steps3) # define the pipeline object.
parameters3 = {'kpca__n_components':[1,2,3,4],
               'kpca__kernel': ['linear', 'rbf','poly'],
               'kpca__gamma': [0.01,0.1,0.5,2,5],
               'lda__n_components':[1,2,3],
               'svm__C':[1, 10, 100,1000],
               'svm__gamma':[0.01,0.001, 0.0001]
               }
grid3 = GridSearchCV(pipeline3, param_grid=parameters3,scoring='accuracy', cv=5)
grid3.fit(x_train, y_train)
# Best accuracy
#print('Best training accuracy: %.3f' % grid3.best_score_)
#training_score_kpca_lda_svm=grid3.best_score_
# Best params
print('\nBest params:\n', grid3.best_params_)
#=====================================================================================================================
# select between the scalers
scaler='standard'
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
x_train_rescaled=x_train_rescaled.iloc[0:x_train_rescaled.shape[0] , 1:5]
kpca = KernelPCA(kernel=grid.best_params_['kpca__kernel'],n_components=grid.best_params_['kpca__n_components'],gamma=grid.best_params_['kpca__gamma'])
# fit and transform x_train_rescaled
start_time_train_kpca_lda_knn = time.perf_counter()
x_train_rescaled_after_kpca = kpca.fit(x_train_rescaled,y_train).transform(x_train_rescaled)
end_time_train_kpca_lda_knn = time.perf_counter()
elapsed_time_train_kpca_lda_knn= end_time_train_kpca_lda_knn - start_time_train_kpca_lda_knn
x_train_rescaled_after_kpca_df=pd.DataFrame(x_train_rescaled_after_kpca,index=x_train.index)
# transform x_test_rescaled
x_test_rescaled=x_test_rescaled.iloc[0:x_test_rescaled.shape[0] , 1:5]
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

y_test=pd.DataFrame(y_test,columns=['Species'],index=x_test.index)
y_train=pd.DataFrame(y_train,columns=['Species'],index=x_train.index)

if x_train_rescaled_after_kpca.shape[1]==1:
    # plot KPCA and LDA training and testing values
    fig2 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Training dataset for model:KPCA-LDA-KNN") 
    plt.scatter(x_train_rescaled_after_kpca[:, 0],x_train_rescaled_after_kpca[:, 0],c=y_train['Species'].values)
    plt.show()
    
    fig3 = plt.figure(figsize = (8,8))
    plt.title("LDA-Training dataset for model:KPCA-LDA-KNN") 
    plt.scatter(x_train_rescaled_after_kpca_and_lda[:, 0],x_train_rescaled_after_kpca_and_lda[:, 0],c=y_train['Species'].values)
    plt.show()
    
    fig4 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Test dataset for model:KPCA-LDA-KNN") 
    plt.scatter(x_test_rescaled_after_kpca[:, 0],x_test_rescaled_after_kpca[:, 0],c=y_test['Species'].values)
    plt.show()
    
    fig5= plt.figure(figsize = (8,8))
    plt.title("LDA-Test dataset for model:KPCA-LDA-KNN") 
    plt.scatter(x_test_rescaled_after_kpca_and_lda[:, 0],x_test_rescaled_after_kpca_and_lda[:, 0],c=y_test['Species'].values)
    plt.show()
if x_train_rescaled_after_kpca.shape[1]==2:
    # plot KPCA and LDA training and testing values
    fig2 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Training dataset for model:KPCA-LDA-KNN") 
    plt.scatter(x_train_rescaled_after_kpca[:, 0],x_train_rescaled_after_kpca[:, 1],c=y_train)
    plt.show()
    
    fig3 = plt.figure(figsize = (8,8))
    plt.title("LDA-Training dataset for model:KPCA-LDA-KNN") 
    plt.scatter(x_train_rescaled_after_kpca_and_lda[:, 0],x_train_rescaled_after_kpca_and_lda[:, 0],c=y_train)
    plt.show()
    
    fig4 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Test dataset for model:KPCA-LDA-KNN") 
    plt.scatter(x_test_rescaled_after_kpca[:, 0],x_test_rescaled_after_kpca[:, 1],c=y_test['Species'].values)
    plt.show()
    
    fig5= plt.figure(figsize = (8,8))
    plt.title("LDA-Test dataset for model:KPCA-LDA-KNN") 
    plt.scatter(x_test_rescaled_after_kpca_and_lda[:, 0],x_test_rescaled_after_kpca_and_lda[:, 0],c=y_test['Species'].values)
    plt.show()
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

if x_train_rescaled_after_kpca2.shape[1]==1:
    # plot KPCA and LDA training and testing values
    fig2 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Training dataset for model:KPCA-LDA-NCC") 
    plt.scatter(x_train_rescaled_after_kpca2[:, 0],x_train_rescaled_after_kpca2[:, 0],c=y_train['Species'].values)
    plt.show()
    
    fig3 = plt.figure(figsize = (8,8))
    plt.title("LDA-Training dataset for model:KPCA-LDA-NCC") 
    plt.scatter(x_train_rescaled_after_kpca_and_lda2[:, 0],x_train_rescaled_after_kpca_and_lda2[:, 0],c=y_train['Species'].values)
    plt.show()
    
    fig4 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Test dataset for model:KPCA-LDA-NCC") 
    plt.scatter(x_test_rescaled_after_kpca2[:, 0],x_test_rescaled_after_kpca2[:, 0],c=y_test['Species'].values)
    plt.show()
    
    fig5= plt.figure(figsize = (8,8))
    plt.title("LDA-Test dataset for model:KPCA-LDA-NCC") 
    plt.scatter(x_test_rescaled_after_kpca_and_lda2[:, 0],x_test_rescaled_after_kpca_and_lda2[:, 0],c=y_test['Species'].values)
    plt.show()
if x_train_rescaled_after_kpca2.shape[1]==2:
    # plot KPCA and LDA training and testing values
    fig2 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Training dataset for model:KPCA-LDA-NCC") 
    plt.scatter(x_train_rescaled_after_kpca2[:, 0],x_train_rescaled_after_kpca2[:, 1],c=y_train['Species'].values)
    plt.show()
    
    fig3 = plt.figure(figsize = (8,8))
    plt.title("LDA-Training dataset for model:KPCA-LDA-NCC") 
    plt.scatter(x_train_rescaled_after_kpca_and_lda2[:, 0],x_train_rescaled_after_kpca_and_lda2[:, 0],c=y_train['Species'].values)
    plt.show()
    
    fig4 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Test dataset for model:KPCA-LDA-NCC") 
    plt.scatter(x_test_rescaled_after_kpca2[:, 0],x_test_rescaled_after_kpca2[:, 1],c=y_test['Species'].values['Species'].values)
    plt.show()
    
    fig5= plt.figure(figsize = (8,8))
    plt.title("LDA-Test dataset for model:KPCA-LDA-NCC") 
    plt.scatter(x_test_rescaled_after_kpca_and_lda2[:, 0],x_test_rescaled_after_kpca_and_lda2[:, 0],c=y_test['Species'].values)
    plt.show()
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

if x_train_rescaled_after_kpca3.shape[1]==1:
    # plot KPCA and LDA training and testing values
    fig2 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Training dataset for model:KPCA-LDA-SVM") 
    plt.scatter(x_train_rescaled_after_kpca3[:, 0],x_train_rescaled_after_kpca3[:, 0],c=y_train['Species'].values)
    plt.show()
    
    fig3 = plt.figure(figsize = (8,8))
    plt.title("LDA-Training dataset for model:KPCA-LDA-SVM") 
    plt.scatter(x_train_rescaled_after_kpca_and_lda3[:, 0],x_train_rescaled_after_kpca_and_lda3[:, 0],c=y_train['Species'].values)
    plt.show()
    
    fig4 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Test dataset for model:KPCA-LDA-SVM") 
    plt.scatter(x_test_rescaled_after_kpca3[:, 0],x_test_rescaled_after_kpca3[:, 0],c=y_test['Species'].values)
    plt.show()
    
    fig5= plt.figure(figsize = (8,8))
    plt.title("LDA-Test dataset for model:KPCA-LDA-SVM") 
    plt.scatter(x_test_rescaled_after_kpca_and_lda3[:, 0],x_test_rescaled_after_kpca_and_lda3[:, 0],c=y_test['Species'].values)
    plt.show()
if x_train_rescaled_after_kpca3.shape[1]==2:
    # plot KPCA and LDA training and testing values
    fig2 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Training dataset for model:KPCA-LDA-SVM") 
    plt.scatter(x_train_rescaled_after_kpca3[:, 0],x_train_rescaled_after_kpca3[:, 1],c=y_train['Species'].values)
    plt.show()
    
    fig3 = plt.figure(figsize = (8,8))
    plt.title("LDA-Training dataset for model:KPCA-LDA-SVM") 
    plt.scatter(x_train_rescaled_after_kpca_and_lda3[:, 0],x_train_rescaled_after_kpca_and_lda3[:, 0],c=y_train['Species'].values)
    plt.show()
    
    fig4 = plt.figure(figsize = (8,8))
    plt.title("Kernel PCA-Test dataset for model:KPCA-LDA-SVM") 
    plt.scatter(x_test_rescaled_after_kpca3[:, 0],x_test_rescaled_after_kpca3[:, 1],c=y_test['Species'].values)
    plt.show()
    
    fig5= plt.figure(figsize = (8,8))
    plt.title("LDA-Test dataset for model:KPCA-LDA-SVM") 
    plt.scatter(x_test_rescaled_after_kpca_and_lda3[:, 0],x_test_rescaled_after_kpca_and_lda3[:, 0],c=y_test['Species'].values)
    plt.show()
#=====================================================================================================================
# compare with the best svm model created in the first project
svm_without_kpca_lda = SVC(kernel='linear', C=100)
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
y_test=pd.DataFrame(y_test,columns=['Species'],index=x_test.index)
y_predicted_svm.columns=['Species']
y_predicted_knn.columns=['Species']
y_predicted_ncc.columns=['Species']
y_predicted_svm_without_kpca_lda.columns=['Species']
def find_if_wrong_class(y_predicted,y_test):
    for i in list(y_predicted.index):
        pred = y_predicted['Species'][i]
        true= y_test['Species'][i]
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
    class_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
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
end_iris = time.perf_counter()
elapsed_iris = end_iris - start_iris
print('The program finished in : '+ str(elapsed_iris)+' seconds.') # 134.10865659999945 seconds.