#=====================================================================================================================
import time
import operator
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,Normalizer
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# START THE OVERALL TIMER
start_iris = time.perf_counter()
#=====================================================================================================================
# LOAD DATA
select_data='Iris.csv'
data=pd.read_csv(select_data)
# DISTINGUISH BETWEEN FEATURES AND TARGETS
rows = data.shape[0]
cols = data.shape[1]
x = data.iloc[0:rows , 0:cols-1]
y = data.iloc[0:rows, cols-1:cols]
# DIVIDE DATA INTO TRAINSET (60%) AND TESTSET (40%)
test_size_percentage=0.4
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=test_size_percentage ,random_state=1)
#=====================================================================================================================
# SCALE THE DATA DEPENDING ON THEIR DISTRIBUTION 
# (PCA is effected by scale so you need to scale the features in your data before applying PCA.)
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
# DIMENTIONALITY REDUCTION WITH PCA
# fitting PCA on the training set only
# check the number of components that keep 90% of the information
pca = PCA(n_components=2)
x_train_rescaled_after_pca = pca.fit_transform(x_train_rescaled)
x_train_rescaled_after_pca_df=pd.DataFrame(x_train_rescaled_after_pca,index=x_train.index)
x_test_rescaled_after_pca = pca.transform(x_test_rescaled)
x_test_rescaled_after_pca_df=pd.DataFrame(x_test_rescaled_after_pca,index=x_test.index)
#=====================================================================================================================
## 3d graph of 3 features
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure(1, figsize=(16, 9))
#ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = PCA(n_components=3).fit_transform(x_train_rescaled)
#ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],cmap=plt.cm.Set1, edgecolor='k', s=40)
#ax.set_title("First three PCA directions")
#ax.set_xlabel("1st eigenvector")
#ax.w_xaxis.set_ticklabels([])
#ax.set_ylabel("2nd eigenvector")
#ax.w_yaxis.set_ticklabels([])
#ax.set_zlabel("3rd eigenvector")
#ax.w_zaxis.set_ticklabels([])
#plt.show()
#print("The number of features in the new subspace is " ,X_reduced.shape[1])
##=====================================================================================================================
### test C and gamma parameters
#from sklearn import datasets
#iris = datasets.load_iris()
#X = iris.data[:, :2] 
#y = iris.target
#def plotSVC(title):
## create a mesh to plot in
#    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#    h = (x_max / x_min)/100
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
#    plt.subplot(1, 1, 1)
#    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
#    plt.xlabel('Sepal length')
#    plt.ylabel('Sepal width')
#    plt.xlim(xx.min(), xx.max())
#    plt.title(title)
#    plt.show()
#
#kernel='rbf'
#gammas_test = [0.1, 1, 10, 100]
#for gamma in gammas_test:
#    C_value=0.1
#    svc = svm.SVC(kernel=kernel,C=C_value,gamma=gamma).fit(X, y)
#    fig=plt.figure(figsize = (8,8))
#    plotSVC('kernel = '+ kernel + ' with C =' +str(C_value) + ' and gamma = '+ str(gamma))
#    
#C_test = [0.1, 1, 10, 100,1000]
#for c in C_test:
#    gamma_value=0.1
#    svc = svm.SVC(kernel=kernel,C=c,gamma=gamma_value).fit(X, y)
#    fig=plt.figure(figsize = (8,8))
#    plotSVC('kernel = '+ kernel + ' with C =' +str(c) + ' and gamma = '+ str(gamma_value))
##=====================================================================================================================
principal_data_frame = pd.DataFrame(data = x_train_rescaled_after_pca, columns = ['PC1', 'PC2'])
final_data_frame = pd.DataFrame(index=principal_data_frame.index)
final_data_frame=y_train
s = np.arange(90)
final_data_frame=final_data_frame.set_index([s])
final_data_frame = pd.concat([principal_data_frame, final_data_frame], axis = 1)
# Explained variation per principal component
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
fig1 = plt.figure(figsize = (8,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
#Together, the two components contain (total_percentage)% of the information.
total_percentage=pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1]
print('Together, the two components contain: ('+ str(total_percentage) +'%) of the initial information.')
# scree plot
fig2 = plt.figure(figsize = (8,8))
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
columns = ['PC1 ('+ str(percent_variance[0])+ '%)', 'PC2 ('+ str(percent_variance[1])+ '%)']
plt.bar(x= np.arange(2), height=percent_variance,width=0.5, tick_label=columns)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Components')
plt.title('PCA Scree Plot')
plt.show()
# plot the principal components on 2D
fig3 = plt.figure(figsize = (8,8))
plt.scatter(principal_data_frame.PC1, principal_data_frame.PC2)
plt.title('PC1 against PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plot
fig4 = plt.figure(figsize = (8,8))
ax = fig4.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = final_data_frame['Species'] == target
    ax.scatter(final_data_frame.loc[indicesToKeep, 'PC1'], final_data_frame.loc[indicesToKeep, 'PC2'], c = color, s = 50)
ax.legend(targets)
ax.grid()
#=====================================================================================================================
# GRID SEARCH FOR FINDING THE BEST KERNEL BETWEEN LINEAR RBF AND SIGMOID
parameter_candidates = [
  {'C': [1, 10, 100,1000], 'kernel': ['linear']},
  {'C': [1, 10, 100,1000], 'gamma': [0.01,0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100,1000], 'gamma': [0.01,0.001, 0.0001], 'kernel': ['sigmoid']}]
scores = ['precision', 'recall','accuracy']
for score in scores:
    print("Performing Grid search for %s...\n" % score)
    clf = GridSearchCV(SVC(), parameter_candidates,cv=5)
    clf.fit(x_train_rescaled, y_train)
    print("THE BEST PARAMETERS OF THE GRID ARE :")
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))    
    y_true, y_pred = y_test,clf.predict(x_test_rescaled)
    y_pred=pd.DataFrame(y_pred,index=y_test.index)
    print("Classification report for classifier %s:\n%s\n"% (clf,classification_report(y_true, y_pred)))
    print("Accuracy={}".format(accuracy_score(y_true, y_pred)))
# DEFINE THE BEST SVM MODEL
best_param= clf.best_params_
select_from_C=best_param['C']
select_from_kernels=best_param['kernel']
if 'gamma' in best_param:
    select_from_gammas=best_param['gamma']
    model = svm.SVC(kernel=select_from_kernels, C=select_from_C, gamma=select_from_gammas,cv=5)
else:
    model = svm.SVC(kernel=select_from_kernels, C=select_from_C)
#=====================================================================================================================
# define a dictionary to keep the temporal scores of each model
time_dict_training= {'svm':'-','knn':'-','ncc':'-'} 
# TRAIN THE BEST SVM MODEL
start_time_train = time.perf_counter()
print('Start svm training at {}'.format(str(start_time_train)))
model.fit(x_train_rescaled_after_pca,y_train.values.ravel())
end_time_train = time.perf_counter()
elapsed_time_train= end_time_train - start_time_train
print('Elapsed training {}'.format(str(elapsed_time_train)))
time_dict_training.update(svm = elapsed_time_train)
start_time_predict = time.perf_counter()
print('Start svm predicting at {}'.format(str(start_time_predict)))
y_predicted_svm = model.predict(x_test_rescaled_after_pca_df)
y_predicted_svm=pd.DataFrame(y_predicted_svm,index=x_test_rescaled_after_pca_df.index)
end_time_predict = time.perf_counter()
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
print('Start knn predicting at {}'.format(str(start_time_predict_knn)))
y_predicted_knn = knn.predict(x_test_rescaled_after_pca_df)
y_predicted_knn=pd.DataFrame(y_predicted_knn,index=x_test_rescaled_after_pca_df.index)
end_time_predict_knn = time.perf_counter()
elapsed_time_predict_knn= end_time_predict_knn - start_time_predict_knn
print('Elapsed knn predicting {}'.format(str(elapsed_time_predict_knn)))
training_score_knn=knn.score(x_train_rescaled_after_pca, y_train)
testing_score_knn=knn.score(x_test_rescaled_after_pca_df, y_test)
#=====================================================================================================================
# NEAREST CLASS CENTROID
ncc = NearestCentroid(metric='euclidean', shrink_threshold=None)
start_time_ncc = time.perf_counter()
print('Start ncc training at {}'.format(str(start_time_ncc)))
ncc.fit(x_train_rescaled_after_pca,y_train.values.ravel())
end_time_ncc = time.perf_counter()
elapsed_time_train_ncc= end_time_ncc - start_time_ncc
print('Elapsed ncc training {}'.format(str(elapsed_time_train_ncc)))
time_dict_training.update(ncc=elapsed_time_train_ncc)
start_time_predict_ncc = time.perf_counter()
print('Start ncc predicting at {}'.format(str(start_time_predict_ncc)))
y_predicted_ncc = ncc.predict(x_test_rescaled_after_pca_df)
y_predicted_ncc=pd.DataFrame(y_predicted_ncc,index=x_test_rescaled_after_pca_df.index)
end_time_predict_ncc = time.perf_counter()
elapsed_time_predict_ncc= end_time_predict_ncc - start_time_predict_ncc
print('Elapsed predicting {}'.format(str(elapsed_time_predict_ncc)))
training_score_ncc=knn.score(x_train_rescaled_after_pca, y_train)
testing_score_ncc=knn.score(x_test_rescaled_after_pca_df, y_test)
#=====================================================================================================================
# SORT SPEED DICTIONARY BETWEEN SVM KNN AND NCC
time_dict_training_sorted=sorted(time_dict_training.items(),key=operator.itemgetter(1))
print('Speed sorted by order after training SVM,KNN and NCC algorithms.')
print('1. Fastest        : '+ str(time_dict_training_sorted[0]) +'.\n2. Of medium speed: '+ str(time_dict_training_sorted[1])+'.\n3. Slowest        : '+ str(time_dict_training_sorted[2])+'.')
#=====================================================================================================================
# SORT ACCURACY DICTIONARY TRAINING BETWEEN SVM KNN AND NCC
accuracy_dict_training= {'svm':'-','knn':'-','ncc':'-'} 
accuracy_dict_training.update(svm=training_score_svm)
accuracy_dict_training.update(knn=training_score_knn)
accuracy_dict_training.update(ncc=training_score_ncc)
accuracy_training_sorted=sorted(accuracy_dict_training.items(),key=operator.itemgetter(1))
print('Accuracy sorted by order after training SVM,KNN and NCC algorithms.')
print('1. Most accurate(training): '+ str(accuracy_training_sorted[2]) +'.\n2. Of medium accuracy(training): '+ str(accuracy_training_sorted[1])+'.\n3. Least accurate(training): '+ str(accuracy_training_sorted[0])+'.')
#=====================================================================================================================
# SORT ACCURACY DICTIONARY TESTING BETWEEN SVM KNN AND NCC
accuracy_dict_testing= {'svm':'-','knn':'-','ncc':'-'} 
accuracy_dict_testing.update(svm=testing_score_svm)
accuracy_dict_testing.update(knn=testing_score_knn)
accuracy_dict_testing.update(ncc=testing_score_ncc)
accuracy_testing_sorted=sorted(accuracy_dict_testing.items(),key=operator.itemgetter(1))
print('Accuracy sorted by order after testing SVM,KNN and NCC algorithms.')
print('1. Most accurate(testing): '+ str(accuracy_training_sorted[2]) +'.\n2. Of medium accuracy(testing): '+ str(accuracy_testing_sorted[1])+'.\n3. Least accurate(testing): '+ str(accuracy_testing_sorted[0])+'.')
#=====================================================================================================================
y_predicted_svm.columns=['Species']
y_predicted_knn.columns=['Species']
y_predicted_ncc.columns=['Species']

def find_if_wrong_class(y_predicted,y_test):
    for i in list(y_predicted.index):
        pred = y_predicted['Species'][i]
        true= y_test['Species'][i]
        if pred != true:
            print("The predicted label is: %s, but the true label is: %s, located in index: %d" % (pred, true,i))
        else:
            print("Correct classification!")
find_if_wrong_class(y_predicted_svm,y_test)
find_if_wrong_class(y_predicted_knn,y_test)
find_if_wrong_class(y_predicted_ncc,y_test)
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
computeAndPlotCM(y_test,y_predicted_svm)
computeAndPlotCM(y_test,y_predicted_knn)
computeAndPlotCM(y_test,y_predicted_ncc)

#=====================================================================================================================
# FIND HOW MUCH TIME HAS ELAPSED
end_iris = time.perf_counter()
elapsed_iris = end_iris - start_iris
print('The program finished in : '+ str(elapsed_iris)+' seconds.') #0.8550779000000048 seconds.