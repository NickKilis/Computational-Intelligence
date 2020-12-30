#=====================================================================================================================
import time
import math
import heapq
import pickle
import operator
import numpy as np
import random as rd
import pandas as pd
import seaborn as sns
from sklearn import metrics
from collections import Counter 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,Normalizer
import warnings
warnings.filterwarnings("ignore")
#=====================================================================================================================
# START THE OVERALL TIMER
start_mnist = time.perf_counter()
#=====================================================================================================================
# LOAD DATA
select_data='Iris.csv'
data=pd.read_csv(select_data)
# DISTINGUISH BETWEEN FEATURES AND TARGETS
rows = data.shape[0]
cols = data.shape[1]
x = data.iloc[0:rows , 0:cols-1]
y = data.iloc[0:rows, cols-1:cols]
# USE LABEL ENCODER TO REPLACE NAMES WITH NUMERICAL VALUES
X = x[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y = y['Species'].values
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) 
label_dict = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2:'Iris-Virginica'}
# VISUALIZE THE DISTRIBUTIONS
#sns.pairplot(data=data,kind='scatter', hue='Species')
# DIVIDE DATA INTO TRAINSET (60%) AND TESTSET (40%)
test_size_percentage=0.4
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=test_size_percentage, stratify=y,random_state=1)
#=====================================================================================================================
# SELECT BETWEEN THE SCALERS BASED ON DATA DISTRIBUTION
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
# EXHAUSTIVE SEARCH FOR FINDING THE BEST COMBINATION OF PARAMETERS FOR LLE AND SPECTRAL CLUSTERING
toggle_search=1 #button for starting the exhaustive search (RUNTIME: 3.190841599999999 seconds.)
if toggle_search==1:
    silhouette_scores_for_diff_No_neighbors=[]
    for num_of_neighbors in range(2,8):
        for num_of_cluster in range(2,8):
            for iterations in [150,200,250]:
                embedding = LocallyLinearEmbedding(n_neighbors=num_of_neighbors, n_components=2,max_iter=iterations )
                x_train_rescaled_LLE = embedding.fit_transform(x_train_rescaled)
                spectral_model_nn = SpectralClustering(n_clusters = num_of_cluster, affinity ='nearest_neighbors')
                labels_nn = spectral_model_nn.fit_predict(x_train_rescaled_LLE) 
                silhouette_avg = silhouette_score(x_train_rescaled_LLE, labels_nn)
                silhouette_scores_for_diff_No_neighbors.append(silhouette_avg)
                print(" For n_neighbors = ",num_of_neighbors,", n_clusters =", num_of_cluster,"and max iterations",iterations,"the average silhouette_score is :", silhouette_avg)
# INTERPRETATION OF RESULTS
ymax = max(silhouette_scores_for_diff_No_neighbors)
xmax = silhouette_scores_for_diff_No_neighbors.index(ymax)
def find_n_largest_values(data,n):
    ten_largest_numbers=heapq.nlargest(n, data)
    ten_largest_positions=[]
    for i in range(0,len(ten_largest_numbers)):
        ten_largest_positions.append(data.index(ten_largest_numbers[i]))
    return(ten_largest_numbers,ten_largest_positions)
ten_largest_numbers_total,ten_largest_positions_total=find_n_largest_values(silhouette_scores_for_diff_No_neighbors,10)
# divide into 3 groups according to the max iterations
s_score_max_iter_150=[silhouette_scores_for_diff_No_neighbors[i] for i in range(0,108,3)]   # index: 0,3,6,9,12,15....
s_score_max_iter_200=[silhouette_scores_for_diff_No_neighbors[i] for i in range(1,108,3)]   # index: 1,4,7,10,13,16...
s_score_max_iter_250=[silhouette_scores_for_diff_No_neighbors[i] for i in range(2,108,3)]   # index: 2,5,8,11,14,17...
ten_largest_numbers_150,ten_largest_positions_150=find_n_largest_values(s_score_max_iter_150,10)
ten_largest_numbers_200,ten_largest_positions_200=find_n_largest_values(s_score_max_iter_200,10)
ten_largest_numbers_250,ten_largest_positions_250=find_n_largest_values(s_score_max_iter_250,10)
matrix_150=np.reshape(s_score_max_iter_150,(6,6))
matrix_200=np.reshape(s_score_max_iter_200,(6,6))
matrix_250=np.reshape(s_score_max_iter_250,(6,6))
def plot_matrix(matrix,title_of_matrix):
    fig = plt.figure(figsize = (8,8))
    plt.title(title_of_matrix)
    Ηeatmap = sns.heatmap(matrix, annot=True, fmt=".3f",annot_kws={'size':16},cmap='coolwarm',linewidths=.5,linecolor='k',mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "horizontal"},cbar=True)
    Ηeatmap.set(xlabel='Number of clusters', ylabel='Number of neighbors')
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()
# PLOT THE RESULTS IN 3 GROUPS
plot_matrix(matrix_150,'Matrix for 150 max iterations')
plot_matrix(matrix_200,'Matrix for 200 max iterations')
plot_matrix(matrix_250,'Matrix for 250 max iterations')
#=====================================================================================================================
# KEEP THE BEST RESULTS
which_19th=[math.ceil(x / 6)+1 for x in ten_largest_positions_150]
which_element_of_19th=[x-(math.floor(x / 6)*6)+1 for x in ten_largest_positions_150]
first_column_150=pd.DataFrame(which_19th,columns=['N_Neighbors_150iter'])
second_column_150=pd.DataFrame(which_element_of_19th,columns=['N_Clusters_150iter'])
third_column_150=pd.DataFrame(ten_largest_numbers_150,columns=['Silhouette_Score_150iter'])
best_10_results_150=first_column_150.join([ second_column_150, third_column_150])

which_19th=[round(x / 6)+1 for x in ten_largest_positions_200]
which_element_of_19th=[x-round(x / 6)*6+1 for x in ten_largest_positions_200]
first_column_200=pd.DataFrame(which_19th,columns=['N_Neighbors_200iter'])
second_column_200=pd.DataFrame(which_element_of_19th,columns=['N_Clusters_200iter'])
third_column_200=pd.DataFrame(ten_largest_numbers_200,columns=['Silhouette_Score_200iter'])
best_10_results_200=first_column_200.join([ second_column_200, third_column_200])

which_19th=[round(x / 6)+1 for x in ten_largest_positions_250]
which_element_of_19th=[x-round(x / 6)*6+1 for x in ten_largest_positions_250]
first_column_250=pd.DataFrame(which_19th,columns=['N_Neighbors_250iter'])
second_column_250=pd.DataFrame(which_element_of_19th,columns=['N_Clusters_250iter'])
third_column_250=pd.DataFrame(ten_largest_numbers_250,columns=['Silhouette_Score_250iter'])
best_10_results_250=first_column_250.join([ second_column_250, third_column_250])
# BEST RESULTS
results=best_10_results_150.join([best_10_results_200,best_10_results_250])
# BEST PARAMETERS FROM RESULTS
N_Neighbors=5
N_Clusters=3
Max_iterations=150
#=====================================================================================================================
# LLE SKLEARN
embedding = LocallyLinearEmbedding(n_neighbors=N_Neighbors, n_components=2,max_iter=Max_iterations )
x_train_rescaled_LLE = embedding.fit_transform(x_train_rescaled)
#=====================================================================================================================
# SPECTRAL CLUSTERING SKLEARN
# Building the clustering model with K-NEAREST NEIGHBORS
spectral_model_nn = SpectralClustering(n_clusters = N_Clusters, affinity ='nearest_neighbors') 
# Training the model and Storing the predicted cluster labels 
labels_nn = spectral_model_nn.fit_predict(x_train_rescaled_LLE) 
silhouette_avg = silhouette_score(x_train_rescaled_LLE, labels_nn)
print('---------------------------------------------------------------------')
print('The silhouette score after Spectral Clustering is:',silhouette_avg)
#=====================================================================================================================
# FIND THE MAJORITY IN EACH CLUSTER
y_train=pd.DataFrame(y_train,columns=['Class'],index=x_train.index)
x_train_rescaled_LLE_df=pd.DataFrame(x_train_rescaled_LLE,index=y_train.index)
labels_nn_df=pd.DataFrame(labels_nn,columns=['Cluster'],index=y_train.index)
cluster0=labels_nn_df[labels_nn_df['Cluster']==0]
cluster1=labels_nn_df[labels_nn_df['Cluster']==1]
cluster2=labels_nn_df[labels_nn_df['Cluster']==2]
xx0=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster0.index)),columns=['Same indices with cluster0']).to_dict()
xx1=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster1.index)),columns=['Same indices with cluster1']).to_dict()
xx2=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster2.index)),columns=['Same indices with cluster2']).to_dict()
all_indices = {}
for d in [xx0,xx1,xx2]:
  all_indices.update(d)
# all_indices['Same indices with class0'][i] # select the i-th element of dict "all_indices" from "Same indices with class0"
coordinates_of_cluster0=pd.DataFrame()
coordinates_of_cluster1=pd.DataFrame()
coordinates_of_cluster2=pd.DataFrame()
for i in range(0,len(all_indices['Same indices with cluster0'])):
    coordinates_of_cluster0=coordinates_of_cluster0.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster0'][i]]])
for i in range(0,len(all_indices['Same indices with cluster1'])):
    coordinates_of_cluster1=coordinates_of_cluster1.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster1'][i]]])    
for i in range(0,len(all_indices['Same indices with cluster2'])):
    coordinates_of_cluster2=coordinates_of_cluster2.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster2'][i]]])
coordinates_of_centroid0=pd.DataFrame(coordinates_of_cluster0.mean(axis=0))
coordinates_of_centroid1=pd.DataFrame(coordinates_of_cluster1.mean(axis=0))
coordinates_of_centroid2=pd.DataFrame(coordinates_of_cluster2.mean(axis=0))
fig = plt.figure(figsize = (8,8))
plt.scatter(x_train_rescaled_LLE[:,0],x_train_rescaled_LLE[:,1],c='gray')
plt.scatter(coordinates_of_centroid0[0][0],coordinates_of_centroid0[0][1],c='red')
plt.scatter(coordinates_of_centroid1[0][0],coordinates_of_centroid1[0][1],c='green')
plt.scatter(coordinates_of_centroid2[0][0],coordinates_of_centroid2[0][1],c='blue')
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.show()
classes_of_cluster0=pd.concat([coordinates_of_cluster0, y_train], axis=1)
classes_of_cluster0=classes_of_cluster0.dropna()
classes_of_cluster1=pd.concat([coordinates_of_cluster1, y_train], axis=1)
classes_of_cluster1=classes_of_cluster1.dropna()
classes_of_cluster2=pd.concat([coordinates_of_cluster2, y_train], axis=1)
classes_of_cluster2=classes_of_cluster2.dropna()
def most_frequent(List): 
    counter = 0
    num = List[0] 
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num
most_freq_class_cluster0=most_frequent(classes_of_cluster0['Class'].tolist())
most_freq_class_cluster1=most_frequent(classes_of_cluster1['Class'].tolist())
most_freq_class_cluster2=most_frequent(classes_of_cluster2['Class'].tolist())
# PRINT THE MOST FREQUENT ELEMENTS
print('---------------------------------------------------------------------')
print('The most frequent class in cluster 0 is :'+str(most_freq_class_cluster0))
print('The most frequent class in cluster 1 is :'+str(most_freq_class_cluster1))
print('The most frequent class in cluster 2 is :'+str(most_freq_class_cluster2))
list0=classes_of_cluster0['Class'].tolist()
list0=np.array(list0,dtype=int)
list0=map(str,list0)
res0 = dict(Counter(i for sub in list0 for i in set(sub))) 
list1=classes_of_cluster1['Class'].tolist()
list1=np.array(list1,dtype=int)
list1=map(str,list1)
res1 = dict(Counter(i for sub in list1 for i in set(sub))) 
list2=classes_of_cluster2['Class'].tolist()
list2=np.array(list2,dtype=int)
list2=map(str,list2)
res2 = dict(Counter(i for sub in list2 for i in set(sub))) 
# PRINT THE LIST OF ELEMENTS' FREQUENCY FOR EACH CLUSTER
print('---------------------------------------------------------------------')
print("The list frequency of elements in cluster0 is : " + str(res0))
print("The list frequency of elements in cluster1 is : " + str(res1)) 
print("The list frequency of elements in cluster2 is : " + str(res2)) 
# PLOT THE DATA IN 2D WITH TRUE LABLES AND WITH PREDICTED LABELS
fig = plt.figure(figsize = (8,8))
with plt.style.context('fivethirtyeight'):       
	plt.title("Data after LLE & Spectral Clustering with true labels")
	plt.scatter(x_train_rescaled_LLE[:, 0], x_train_rescaled_LLE[:, 1], c=y_train['Class'], s=30, cmap=plt.cm.get_cmap("rainbow", N_Clusters))
	plt.colorbar(ticks=range(10))
plt.show()
fig = plt.figure(figsize = (8,8))
with plt.style.context('fivethirtyeight'):       
	plt.title("Data after LLE & spectral clustering with predicted labels")
	plt.scatter(x_train_rescaled_LLE[:, 0], x_train_rescaled_LLE[:, 1], c=labels_nn, s=30, cmap=plt.cm.get_cmap("rainbow", N_Clusters))
	plt.colorbar(ticks=range(10))
plt.show()
#=====================================================================================================================
# RUN THE SAME MODEL 100 TIMES TO SEE IF IT HAS CHANGES IN SILHOUETTE SCORE
def test_for_100_diff_initializations(x_train_rescaled_LLE,N_Clusters):
    results_after_100_runs=[]
    for i in range(0,99):
        spectral_model_nn = SpectralClustering(n_clusters = N_Clusters, affinity ='nearest_neighbors') 
        labels_nn = spectral_model_nn.fit_predict(x_train_rescaled_LLE) 
        silhouette_avg = silhouette_score(x_train_rescaled_LLE, labels_nn)
        results_after_100_runs.append(silhouette_avg)
    # INTERPRETATION OF THE RESULTS
    print('INTERPRETATION OF THE RESULTS :')
    if np.mean(results_after_100_runs)>=0.71:
        print('silhouette_score = 0.71-1.0 : A strong structure has been found.')
    if np.mean(results_after_100_runs)>=0.51 and np.mean(results_after_100_runs)<0.70:
        print('silhouette_score = 0.51-0.70: A reasonable structure has been found.')
    if np.mean(results_after_100_runs)>=0.26 and np.mean(results_after_100_runs)<0.50:
        print('silhouette_score = 0.26-0.50: The structure is weak and could be artificial. Try additional methods of data analysis.')
    if np.mean(results_after_100_runs)<0.25:
        print('silhouette_score < 0.25     : No substantial structure has been found.')
    return(results_after_100_runs)
# KEEP THE RESULTS OF 100 RUNS IN A DATAFRAME
results_after_100_runs=test_for_100_diff_initializations(x_train_rescaled_LLE,N_Clusters)
print('---------------------------------------------------------------------')
print("The mean silhouette score after 100 runs is : " , np.mean(results_after_100_runs))
# FIND THE PERCENTAGE OF CORRECT CLUSTERS FOR THE TRAIN DATASET
def find_if_wrong_class(y_predicted,y_test):
    wrong_pred=[]
    wrong_true=[]
    correct_pred=[]
    correct_true=[]
    for i in list(y_predicted.index):
        pred = y_predicted['Class'][i]
        true= y_test['Class'][i]
        if pred != true:
            print("The predicted label is: %s, but the true label is: %s, located in index: %d" % (pred, true,i))
            #break
            wrong_pred.append(pred)
            wrong_true.append(true)
        else:
            print("Correct classification!")
            correct_pred.append(pred)
            correct_true.append(true)
    wrong=pd.DataFrame(zip(wrong_pred, wrong_true))
    correct=pd.DataFrame(zip(correct_pred, correct_true))
    return(wrong,correct)
# LISTS WITH WRONG AND CORRECT LABELS
labels_nn_df=pd.DataFrame(labels_nn,columns=['Class'],index=y_train.index)
(wrong,correct)=find_if_wrong_class(labels_nn_df,y_train)
score_accuracy_train=len(correct)/(len(wrong)+len(correct))*100
print('---------------------------------------------------------------------')
print('The accuracy score in the train dataset is:',score_accuracy_train)
# PLOT THE CONFUSION MATRIX
def computeAndPlotCM(y_test,y_predicted):
    cm=confusion_matrix(y_test,y_predicted)
    def get_df_name(df):
        name =[x for x in globals() if globals()[x] is df][0]
        return name
    name=(get_df_name(y_predicted))
    class_names=['0','1','2']
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
cm_train=computeAndPlotCM(y_train,labels_nn_df)
#=====================================================================================================================
# FIND THE PERCENTAGE OF CORRECT CLUSTERS FOR THE TEST DATASET
x_test_rescaled_LLE = embedding.transform(x_test_rescaled)
y_test=pd.DataFrame(y_test,columns=['Class'],index=x_test.index)
# show the test dataset with centroids from train dataset
fig = plt.figure(figsize = (8,8))
plt.scatter(x_test_rescaled_LLE[:,0],x_test_rescaled_LLE[:,1],c='gray')
plt.scatter(coordinates_of_centroid0[0][0],coordinates_of_centroid0[0][1],c='red')
plt.scatter(coordinates_of_centroid1[0][0],coordinates_of_centroid1[0][1],c='green')
plt.scatter(coordinates_of_centroid2[0][0],coordinates_of_centroid2[0][1],c='blue')
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.show()
# FIND CENTROIDS
centers=np.array([ [coordinates_of_centroid0[0][0],coordinates_of_centroid0[0][1]],
                   [coordinates_of_centroid1[0][0],coordinates_of_centroid1[0][1]],
                   [coordinates_of_centroid2[0][0],coordinates_of_centroid2[0][1]]])
# CREATE VORONOI REGIONS AND PLOT THE DATA WITH THEM
vor = Voronoi(centers) 
voronoi_plot_2d(vor,line_width=0.5,line_colors='r',line_alpha=0.3) 
plt.scatter(x_test_rescaled_LLE[:, 0], x_test_rescaled_LLE[:, 1], c=y_test['Class'], s=5, cmap='rainbow') 
plt.show()
# USE KMEANS TO GROUP THE TEST DATASET ACCORDING TO THE CENTROIDS
kmeans = KMeans(n_clusters=N_Clusters, init='random', n_init=1, random_state=0, max_iter=1000)
kmeans.fit(x_test_rescaled_LLE)
y_kmeans = kmeans.predict(x_test_rescaled_LLE)
# cluster index for each observation
centers2 = kmeans.cluster_centers_ 
# cluster center coordinates
vor2 = Voronoi(centers2)
# PLOT THE TEST DATASET CATEGORISED
voronoi_plot_2d(vor2,line_width=0.5,line_colors='r',line_alpha=0.3) 
plt.scatter(x_test_rescaled_LLE[:, 0], x_test_rescaled_LLE[:, 1], c=y_kmeans, s=5, cmap='rainbow') 
plt.show()
# CONFUSION MATRIX AND ACCURACY FOR THE TES DATASET
cm_test=computeAndPlotCM(y_test,y_kmeans)
(wrong2,correct2)=find_if_wrong_class(pd.DataFrame(y_kmeans,columns=['Class'],index=y_test.index),y_test)
score_accuracy_test=len(correct2)/(len(wrong2)+len(correct2))*100
print('---------------------------------------------------------------------')
print('The accuracy for the test dataset is:',score_accuracy_test)
#=====================================================================================================================
# FIND HOW MUCH TIME HAS ELAPSED
end_mnist = time.perf_counter()
elapsed_mnist = end_mnist - start_mnist
print('The program finished in : '+ str(elapsed_mnist)+' seconds.') # 5.858391600000687 seconds.
#=====================================================================================================================