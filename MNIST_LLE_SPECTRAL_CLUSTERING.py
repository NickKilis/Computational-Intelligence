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
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KNeighborsClassifier
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
 
# create a temporary dataframe that has as a last column the labels
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

# create a temporary dataframe that has as a last column the labels
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
# SCALE THE DATA DEPENDING ON THEIR DISTRIBUTION 
# PCA is effected by scale so we need to scale the features in your data before applying PCA.
test_size_percentage=(len(x_test)/len(x_train))*100
select_data='MNIST'
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
# EXHAUSTIVE SEARCH FOR FINDING THE BEST COMBINATION OF PARAMETERS FOR LLE AND SPECTRAL CLUSTERING
toggle_search=0 #button for starting the exhaustive search
if toggle_search==1:
    silhouette_scores_for_diff_No_neighbors=[]
    for num_of_neighbors in range(2,21):
        for num_of_cluster in range(2,21):
            for iterations in [150,200,250]:
                embedding = LocallyLinearEmbedding(n_neighbors=num_of_neighbors, n_components=2,max_iter=iterations )
                x_train_rescaled_LLE = embedding.fit_transform(x_train_rescaled)
                spectral_model_nn = SpectralClustering(n_clusters = num_of_cluster, affinity ='nearest_neighbors')
                labels_nn = spectral_model_nn.fit_predict(x_train_rescaled_LLE) 
                silhouette_avg = silhouette_score(x_train_rescaled_LLE, labels_nn)
                silhouette_scores_for_diff_No_neighbors.append(silhouette_avg)
                print(" For n_neighbors = ",num_of_neighbors,", n_clusters =", num_of_cluster,"and max iterations",iterations,"the average silhouette_score is :", silhouette_avg)
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
s_score_max_iter_150=[silhouette_scores_for_diff_No_neighbors[i] for i in range(0,1083,3)]   # index: 0,3,6,9,12,15....
s_score_max_iter_200=[silhouette_scores_for_diff_No_neighbors[i] for i in range(1,1083,3)]   # index: 1,4,7,10,13,16...
s_score_max_iter_250=[silhouette_scores_for_diff_No_neighbors[i] for i in range(2,1083,3)]   # index: 2,5,8,11,14,17...
ten_largest_numbers_150,ten_largest_positions_150=find_n_largest_values(s_score_max_iter_150,10)
ten_largest_numbers_200,ten_largest_positions_200=find_n_largest_values(s_score_max_iter_200,10)
ten_largest_numbers_250,ten_largest_positions_250=find_n_largest_values(s_score_max_iter_250,10)
matrix_150=np.reshape(s_score_max_iter_150,(19,19))
matrix_200=np.reshape(s_score_max_iter_200,(19,19))
matrix_250=np.reshape(s_score_max_iter_250,(19,19))
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
which_19th=[math.ceil(x / 19)+1 for x in ten_largest_positions_150]
which_element_of_19th=[x-(math.floor(x / 19)*19)+1 for x in ten_largest_positions_150]
first_column_150=pd.DataFrame(which_19th,columns=['N_Neighbors_150iter'])
second_column_150=pd.DataFrame(which_element_of_19th,columns=['N_Clusters_150iter'])
third_column_150=pd.DataFrame(ten_largest_numbers_150,columns=['Silhouette_Score_150iter'])
best_10_results_150=first_column_150.join([ second_column_150, third_column_150])

which_19th=[round(x / 19)+1 for x in ten_largest_positions_200]
which_element_of_19th=[x-round(x / 19)*19+1 for x in ten_largest_positions_200]
first_column_200=pd.DataFrame(which_19th,columns=['N_Neighbors_200iter'])
second_column_200=pd.DataFrame(which_element_of_19th,columns=['N_Clusters_200iter'])
third_column_200=pd.DataFrame(ten_largest_numbers_200,columns=['Silhouette_Score_200iter'])
best_10_results_200=first_column_200.join([ second_column_200, third_column_200])

which_19th=[round(x / 19)+1 for x in ten_largest_positions_250]
which_element_of_19th=[x-round(x / 19)*19+1 for x in ten_largest_positions_250]
first_column_250=pd.DataFrame(which_19th,columns=['N_Neighbors_250iter'])
second_column_250=pd.DataFrame(which_element_of_19th,columns=['N_Clusters_250iter'])
third_column_250=pd.DataFrame(ten_largest_numbers_250,columns=['Silhouette_Score_250iter'])
best_10_results_250=first_column_250.join([ second_column_250, third_column_250])
# BEST RESULTS STORED IN A DATAFRAME
results=best_10_results_150.join([best_10_results_200,best_10_results_250])
def find_best_from_results():
    a=results['N_Clusters_150iter'].max()
    b=results['N_Clusters_200iter'].max()
    c=results['N_Clusters_250iter'].max()
    d=results.iloc[results['N_Clusters_150iter'].idxmax()]['Silhouette_Score_150iter']
    e=results.iloc[results['N_Clusters_200iter'].idxmax()]['Silhouette_Score_200iter']
    f=results.iloc[results['N_Clusters_250iter'].idxmax()]['Silhouette_Score_250iter']
    g=results.iloc[results['N_Clusters_150iter'].idxmax()]['N_Neighbors_150iter']
    h=results.iloc[results['N_Clusters_200iter'].idxmax()]['N_Neighbors_200iter']
    i=results.iloc[results['N_Clusters_250iter'].idxmax()]['N_Neighbors_250iter']
    def maximum(a,b,c,d,e,f): 
        if (a > b) and (a > c): 
            largest = a
            print('Best:N_Neighbors '+ str(g)+',N_Clusters '+ str(a)+',Max iterations 150.')
        elif (b > a) and (b > c): 
            largest = b
            print('Best:N_Neighbors '+ str(h)+',N_Clusters '+ str(b)+',Max iterations 200.')
        elif (c > a) and (c > b): 
            largest = c
            print('Best:N_Neighbors '+ str(i)+',N_Clusters '+ str(c)+',Max iterations 200.')
        elif a == b and d > e: 
            largest = a
            print('Best:N_Neighbors '+ str(g)+',N_Clusters '+ str(a)+',Max iterations 150.')
        elif a == b and d < e: 
            largest = b
            print('Best:N_Neighbors '+ str(h)+',N_Clusters '+ str(b)+',Max iterations 200.')
        elif a == c and d > f: 
            largest = a
            print('Best:N_Neighbors '+ str(g)+',N_Clusters '+ str(a)+',Max iterations 150.')
        elif a == c and d < f: 
            largest = c
            print('Best:N_Neighbors '+ str(i)+',N_Clusters '+ str(c)+',Max iterations 200.')
        elif c == b and f > e: 
            largest = c
            print('Best:N_Neighbors '+ str(i)+',N_Clusters '+ str(c)+',Max iterations 200.')
        elif c == b and f < e: 
            largest = b
            print('Best:N_Neighbors '+ str(h)+',N_Clusters '+ str(b)+',Max iterations 200.')
        return largest 
    maximum(a,b,c,d,e,f)
# RUN THE FUNCTION TO SEE WHICH PARAMETERS TO USE
find_best_from_results()
# USE THESE PARAMETERS FOR THE BEST MODEL
N_Neighbors=3
N_Clusters=10
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
print('The silhouette score after Spectral Clustering is:',silhouette_avg)
#=====================================================================================================================
# FIND THE MAJORITY IN EACH CLUSTER
x_train_rescaled_LLE_df=pd.DataFrame(x_train_rescaled_LLE,index=y_train.index)
labels_nn_df=pd.DataFrame(labels_nn,columns=['Cluster'],index=y_train.index)
cluster0=labels_nn_df[labels_nn_df['Cluster']==0]
cluster1=labels_nn_df[labels_nn_df['Cluster']==1]
cluster2=labels_nn_df[labels_nn_df['Cluster']==2]
cluster3=labels_nn_df[labels_nn_df['Cluster']==3]
cluster4=labels_nn_df[labels_nn_df['Cluster']==4]
cluster5=labels_nn_df[labels_nn_df['Cluster']==5]
cluster6=labels_nn_df[labels_nn_df['Cluster']==6]
cluster7=labels_nn_df[labels_nn_df['Cluster']==7]
cluster8=labels_nn_df[labels_nn_df['Cluster']==8]
cluster9=labels_nn_df[labels_nn_df['Cluster']==9]
xx0=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster0.index)),columns=['Same indices with cluster0']).to_dict()
xx1=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster1.index)),columns=['Same indices with cluster1']).to_dict()
xx2=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster2.index)),columns=['Same indices with cluster2']).to_dict()
xx3=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster3.index)),columns=['Same indices with cluster3']).to_dict()
xx4=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster4.index)),columns=['Same indices with cluster4']).to_dict()
xx5=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster5.index)),columns=['Same indices with cluster5']).to_dict()
xx6=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster6.index)),columns=['Same indices with cluster6']).to_dict()
xx7=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster7.index)),columns=['Same indices with cluster7']).to_dict()
xx8=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster8.index)),columns=['Same indices with cluster8']).to_dict()
xx9=pd.DataFrame(set(list(x_train_rescaled_LLE_df.index)) & set(list(cluster9.index)),columns=['Same indices with cluster9']).to_dict()
all_indices = {}
for d in [xx0,xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9]:
  all_indices.update(d)
# all_indices['Same indices with class0'][i] # select the i-th element of dict "all_indices" from "Same indices with class0"
coordinates_of_cluster0=pd.DataFrame()
coordinates_of_cluster1=pd.DataFrame()
coordinates_of_cluster2=pd.DataFrame()
coordinates_of_cluster3=pd.DataFrame()
coordinates_of_cluster4=pd.DataFrame()
coordinates_of_cluster5=pd.DataFrame()
coordinates_of_cluster6=pd.DataFrame()
coordinates_of_cluster7=pd.DataFrame()
coordinates_of_cluster8=pd.DataFrame()
coordinates_of_cluster9=pd.DataFrame()
for i in range(0,len(all_indices['Same indices with cluster0'])):
    coordinates_of_cluster0=coordinates_of_cluster0.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster0'][i]]])
for i in range(0,len(all_indices['Same indices with cluster1'])):
    coordinates_of_cluster1=coordinates_of_cluster1.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster1'][i]]])    
for i in range(0,len(all_indices['Same indices with cluster2'])):
    coordinates_of_cluster2=coordinates_of_cluster2.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster2'][i]]])
for i in range(0,len(all_indices['Same indices with cluster3'])):
    coordinates_of_cluster3=coordinates_of_cluster3.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster3'][i]]])
for i in range(0,len(all_indices['Same indices with cluster4'])):
    coordinates_of_cluster4=coordinates_of_cluster4.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster4'][i]]])    
for i in range(0,len(all_indices['Same indices with cluster5'])):
    coordinates_of_cluster5=coordinates_of_cluster5.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster5'][i]]])    
for i in range(0,len(all_indices['Same indices with cluster6'])):
    coordinates_of_cluster6=coordinates_of_cluster6.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster6'][i]]])    
for i in range(0,len(all_indices['Same indices with cluster7'])):
    coordinates_of_cluster7=coordinates_of_cluster7.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster7'][i]]])    
for i in range(0,len(all_indices['Same indices with cluster8'])):
    coordinates_of_cluster8=coordinates_of_cluster8.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster8'][i]]])    
for i in range(0,len(all_indices['Same indices with cluster9'])):
    coordinates_of_cluster9=coordinates_of_cluster9.append([x_train_rescaled_LLE_df.ix[all_indices['Same indices with cluster9'][i]]])        
coordinates_of_centroid0=pd.DataFrame(coordinates_of_cluster0.mean(axis=0))
coordinates_of_centroid1=pd.DataFrame(coordinates_of_cluster1.mean(axis=0))
coordinates_of_centroid2=pd.DataFrame(coordinates_of_cluster2.mean(axis=0))
coordinates_of_centroid3=pd.DataFrame(coordinates_of_cluster3.mean(axis=0))
coordinates_of_centroid4=pd.DataFrame(coordinates_of_cluster4.mean(axis=0))
coordinates_of_centroid5=pd.DataFrame(coordinates_of_cluster5.mean(axis=0))
coordinates_of_centroid6=pd.DataFrame(coordinates_of_cluster6.mean(axis=0))
coordinates_of_centroid7=pd.DataFrame(coordinates_of_cluster7.mean(axis=0))
coordinates_of_centroid8=pd.DataFrame(coordinates_of_cluster8.mean(axis=0))
coordinates_of_centroid9=pd.DataFrame(coordinates_of_cluster9.mean(axis=0))
fig = plt.figure(figsize = (8,8))
plt.scatter(x_train_rescaled_LLE[:,0],x_train_rescaled_LLE[:,1],c='gray')
plt.scatter(coordinates_of_centroid0[0][0],coordinates_of_centroid0[0][1],c='red')
plt.scatter(coordinates_of_centroid1[0][0],coordinates_of_centroid1[0][1],c='green')
plt.scatter(coordinates_of_centroid2[0][0],coordinates_of_centroid2[0][1],c='blue')
plt.scatter(coordinates_of_centroid3[0][0],coordinates_of_centroid3[0][1],c='orange')
plt.scatter(coordinates_of_centroid4[0][0],coordinates_of_centroid4[0][1],c='pink')
plt.scatter(coordinates_of_centroid5[0][0],coordinates_of_centroid5[0][1],c='black')
plt.scatter(coordinates_of_centroid6[0][0],coordinates_of_centroid5[0][1],c='magenta')
plt.scatter(coordinates_of_centroid7[0][0],coordinates_of_centroid5[0][1],c='yellow')
plt.scatter(coordinates_of_centroid8[0][0],coordinates_of_centroid5[0][1],c='cyan')
plt.scatter(coordinates_of_centroid9[0][0],coordinates_of_centroid5[0][1],c='brown')
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.show()
classes_of_cluster0=pd.concat([coordinates_of_cluster0, y_train], axis=1)
classes_of_cluster0=classes_of_cluster0.dropna()
classes_of_cluster1=pd.concat([coordinates_of_cluster1, y_train], axis=1)
classes_of_cluster1=classes_of_cluster1.dropna()
classes_of_cluster2=pd.concat([coordinates_of_cluster2, y_train], axis=1)
classes_of_cluster2=classes_of_cluster2.dropna()
classes_of_cluster3=pd.concat([coordinates_of_cluster3, y_train], axis=1)
classes_of_cluster3=classes_of_cluster3.dropna()
classes_of_cluster4=pd.concat([coordinates_of_cluster4, y_train], axis=1)
classes_of_cluster4=classes_of_cluster4.dropna()
classes_of_cluster5=pd.concat([coordinates_of_cluster5, y_train], axis=1)
classes_of_cluster5=classes_of_cluster5.dropna()
classes_of_cluster6=pd.concat([coordinates_of_cluster6, y_train], axis=1)
classes_of_cluster6=classes_of_cluster6.dropna()
classes_of_cluster7=pd.concat([coordinates_of_cluster7, y_train], axis=1)
classes_of_cluster7=classes_of_cluster7.dropna()
classes_of_cluster8=pd.concat([coordinates_of_cluster8, y_train], axis=1)
classes_of_cluster8=classes_of_cluster8.dropna()
classes_of_cluster9=pd.concat([coordinates_of_cluster9, y_train], axis=1)
classes_of_cluster9=classes_of_cluster9.dropna()
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
most_freq_class_cluster3=most_frequent(classes_of_cluster3['Class'].tolist())
most_freq_class_cluster4=most_frequent(classes_of_cluster4['Class'].tolist())
most_freq_class_cluster5=most_frequent(classes_of_cluster5['Class'].tolist())
most_freq_class_cluster6=most_frequent(classes_of_cluster6['Class'].tolist())
most_freq_class_cluster7=most_frequent(classes_of_cluster7['Class'].tolist())
most_freq_class_cluster8=most_frequent(classes_of_cluster8['Class'].tolist())
most_freq_class_cluster9=most_frequent(classes_of_cluster9['Class'].tolist())
print('The most frequent class in cluster 0 is :'+str(most_freq_class_cluster0))
print('The most frequent class in cluster 1 is :'+str(most_freq_class_cluster1))
print('The most frequent class in cluster 2 is :'+str(most_freq_class_cluster2))
print('The most frequent class in cluster 3 is :'+str(most_freq_class_cluster3))
print('The most frequent class in cluster 4 is :'+str(most_freq_class_cluster4))
print('The most frequent class in cluster 5 is :'+str(most_freq_class_cluster5))
print('The most frequent class in cluster 6 is :'+str(most_freq_class_cluster6))
print('The most frequent class in cluster 7 is :'+str(most_freq_class_cluster7))
print('The most frequent class in cluster 8 is :'+str(most_freq_class_cluster8))
print('The most frequent class in cluster 9 is :'+str(most_freq_class_cluster9))
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
list3=classes_of_cluster3['Class'].tolist()
list3=np.array(list3,dtype=int)
list3=map(str,list3)
res3 = dict(Counter(i for sub in list3 for i in set(sub))) 
list4=classes_of_cluster4['Class'].tolist()
list4=np.array(list4,dtype=int)
list4=map(str,list4)
res4 = dict(Counter(i for sub in list4 for i in set(sub))) 
list5=classes_of_cluster5['Class'].tolist()
list5=np.array(list5,dtype=int)
list5=map(str,list5)
res5 = dict(Counter(i for sub in list5 for i in set(sub))) 
list6=classes_of_cluster6['Class'].tolist()
list6=np.array(list6,dtype=int)
list6=map(str,list6)
res6 = dict(Counter(i for sub in list6 for i in set(sub))) 
list7=classes_of_cluster7['Class'].tolist()
list7=np.array(list7,dtype=int)
list7=map(str,list7)
res7 = dict(Counter(i for sub in list7 for i in set(sub))) 
list8=classes_of_cluster8['Class'].tolist()
list8=np.array(list8,dtype=int)
list8=map(str,list8)
res8 = dict(Counter(i for sub in list8 for i in set(sub))) 
list9=classes_of_cluster9['Class'].tolist()
list9=np.array(list9,dtype=int)
list9=map(str,list9)
res9 = dict(Counter(i for sub in list9 for i in set(sub)))
# PRINT THE LIST OF ELEMENTS' FREQUENCY FOR EACH CLUSTER
print("The list frequency of elements in cluster0 is : " + str(res0))
print("The list frequency of elements in cluster1 is : " + str(res1)) 
print("The list frequency of elements in cluster2 is : " + str(res2)) 
print("The list frequency of elements in cluster3 is : " + str(res3))
print("The list frequency of elements in cluster4 is : " + str(res4))
print("The list frequency of elements in cluster5 is : " + str(res5))
print("The list frequency of elements in cluster6 is : " + str(res6))
print("The list frequency of elements in cluster7 is : " + str(res7))
print("The list frequency of elements in cluster8 is : " + str(res8))
print("The list frequency of elements in cluster9 is : " + str(res9))
# PLOT THE DATA IN 2D WITH TRUE LABLES AND WITH PREDICTED LABELS
fig = plt.figure(figsize = (8,8))
with plt.style.context('fivethirtyeight'):       
	plt.title("Data after LLE & Spectral Clustering with true labels")
	plt.scatter(x_train_rescaled_LLE[:, 0], x_train_rescaled_LLE[:, 1], c=y_train['Class'], s=30, cmap=plt.cm.get_cmap("rainbow", 10))
	plt.colorbar(ticks=range(10))
plt.show()
fig = plt.figure(figsize = (8,8))
with plt.style.context('fivethirtyeight'):       
	plt.title("Data after LLE & spectral clustering with predicted labels")
	plt.scatter(x_train_rescaled_LLE[:, 0], x_train_rescaled_LLE[:, 1], c=labels_nn, s=30, cmap=plt.cm.get_cmap("rainbow", 10))
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
# FIND THE PERCENTAGE OF CORRECT CLUSTERS FOR THE TRAIN DATASET
labels_nn_df=pd.DataFrame(labels_nn,columns=['Class'],index=y_train.index)
# CREATE A FUNCTION THAT RETURNS THE CORRECT AND WRONG CLASSIFIED POINTS
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
(wrong,correct)=find_if_wrong_class(labels_nn_df,y_train)
score_accuracy_train=len(correct)/(len(wrong)+len(correct))*100
print('The accuracy score in the train dataset is:',score_accuracy_train)
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
cm_train=computeAndPlotCM(y_train,labels_nn_df)
# FIND THE PERCENTAGE OF CORRECT CLUSTERS FOR THE TEST DATASET
x_test_rescaled_LLE = embedding.transform(x_test_rescaled)
# show the test dataset with centroids from train dataset
fig = plt.figure(figsize = (8,8))
plt.scatter(x_test_rescaled_LLE[:,0],x_test_rescaled_LLE[:,1],c='gray')
plt.scatter(coordinates_of_centroid0[0][0],coordinates_of_centroid0[0][1],c='red')
plt.scatter(coordinates_of_centroid1[0][0],coordinates_of_centroid1[0][1],c='green')
plt.scatter(coordinates_of_centroid2[0][0],coordinates_of_centroid2[0][1],c='blue')
plt.scatter(coordinates_of_centroid3[0][0],coordinates_of_centroid3[0][1],c='orange')
plt.scatter(coordinates_of_centroid4[0][0],coordinates_of_centroid4[0][1],c='pink')
plt.scatter(coordinates_of_centroid5[0][0],coordinates_of_centroid5[0][1],c='black')
plt.scatter(coordinates_of_centroid6[0][0],coordinates_of_centroid5[0][1],c='magenta')
plt.scatter(coordinates_of_centroid7[0][0],coordinates_of_centroid5[0][1],c='yellow')
plt.scatter(coordinates_of_centroid8[0][0],coordinates_of_centroid5[0][1],c='cyan')
plt.scatter(coordinates_of_centroid9[0][0],coordinates_of_centroid5[0][1],c='brown')
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.show()
# FIND CENTROIDS
centers=np.array([ [coordinates_of_centroid0[0][0],coordinates_of_centroid0[0][1]],
                   [coordinates_of_centroid1[0][0],coordinates_of_centroid1[0][1]],
                   [coordinates_of_centroid2[0][0],coordinates_of_centroid2[0][1]],
                   [coordinates_of_centroid3[0][0],coordinates_of_centroid3[0][1]],
                   [coordinates_of_centroid4[0][0],coordinates_of_centroid4[0][1]],
                   [coordinates_of_centroid5[0][0],coordinates_of_centroid5[0][1]],
                   [coordinates_of_centroid6[0][0],coordinates_of_centroid6[0][1]],
                   [coordinates_of_centroid7[0][0],coordinates_of_centroid7[0][1]],
                   [coordinates_of_centroid8[0][0],coordinates_of_centroid8[0][1]],
                   [coordinates_of_centroid9[0][0],coordinates_of_centroid9[0][1]]])
# CREATE VORONOI REGIONS AND PLOT THE DATA WITH THEM
vor = Voronoi(centers) 
voronoi_plot_2d(vor,line_width=0.5,line_colors='r',line_alpha=0.3) 
plt.scatter(x_test_rescaled_LLE[:, 0], x_test_rescaled_LLE[:, 1], c=y_test['Class'], s=0.5, cmap='rainbow') 
plt.show()
# USE KMEANS TO GROUP THE TEST DATASET ACCORDING TO THE CENTROIDS
kmeans = KMeans(n_clusters=10, init='random', n_init=1, random_state=0, max_iter=1000)
kmeans.fit(x_test_rescaled_LLE)
y_kmeans = kmeans.predict(x_test_rescaled_LLE)
# cluster index for each observation
centers2 = kmeans.cluster_centers_ 
# cluster center coordinates 
# PLOT THE TEST DATASET CATEGORISED
vor2 = Voronoi(centers2) 
voronoi_plot_2d(vor2,line_width=0.5,line_colors='r',line_alpha=0.3) 
plt.scatter(x_test_rescaled_LLE[:, 0], x_test_rescaled_LLE[:, 1], c=y_kmeans, s=0.5, cmap='rainbow') 
plt.show()
# CONFUSION MATRIX AND ACCURACY FOR THE TES DATASET
cm_test=computeAndPlotCM(y_test,y_kmeans)
(wrong2,correct2)=find_if_wrong_class(pd.DataFrame(y_kmeans,columns=['Class'],index=y_test.index),y_test)
score_accuracy_test=len(correct2)/len(wrong2)*100
#=====================================================================================================================
# FIND HOW MUCH TIME HAS ELAPSED
end_mnist = time.perf_counter()
elapsed_mnist = end_mnist - start_mnist
print('The program finished in : '+ str(elapsed_mnist)+' seconds.')
#=====================================================================================================================