#Name - yogesh
#Roll No. - B19273


#importing the neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist 
from sklearn import metrics

#Reading the csv files
train_data=pd.read_csv('mnist-tsne-train.csv')
test_data=pd.read_csv('mnist-tsne-test.csv')

print("-------------------------------Question 1-----------------------------\n")
def purity_score(y_true, y_pred):
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred) #compute confusion matrix
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix) #finding the optimal one-to-one mapping between the cluster labels and true labels
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix) #returning cluster accuracy


true_train=train_data['labels'] # KMeans taking the labels 
true_test=test_data['labels']
train_data.drop(['labels'],axis=1,inplace=True) #drop the attribute labels
test_data.drop(['labels'],axis=1,inplace=True)
X=train_data.iloc[:, [0,1]].values #taking the values of thedata
Y=test_data.iloc[:, [0,1]].values
K = 10
kmeans = KMeans(n_clusters=K,random_state=42) #Making the KMeans model
kmeans.fit(train_data) #Fitting the training data
kmeans_prediction = kmeans.predict(train_data) #Predicting the data
centers=kmeans.cluster_centers_ #Getting the clusters centers
def clusters_plot(k,X,prediction,centers): #Cluster plotting function
    k0=plt.scatter(X[:, 0], X[:, 1], c=prediction,cmap='rainbow') #Plotting the clusters
    plt.colorbar(k0)
    plt.scatter(centers[:, 0], centers[:, 1],  c='black', label = 'Centroids') #Plotting cluster centers
    plt.title('clusters when k ='+str(k))
    plt.xlabel('Dimention-1')
    plt.ylabel('Dimension-2')
    plt.show()
clusters_plot(K,X,kmeans_prediction,centers) #Plotting the training data
print('The purity score of train data by KMeans clustering (K =',K,'):',purity_score(true_train,kmeans_prediction))
kmeans_prediction_test=kmeans.predict(test_data) #Predicting the test data
clusters_plot(K,Y, kmeans_prediction_test, centers) #Plotting the test data
print('The purity score of test data by KMeans clustering (K =',K,'):',purity_score(true_test,kmeans_prediction_test))   

print("-------------------------------Question 2-----------------------------\n")
K = 10
gmm = GaussianMixture(n_components = K,random_state=42) #Preparing GMM model
gmm.fit(train_data) #Fitting training data
GMM_prediction = gmm.predict(train_data) #Predicting the training data
means_train=gmm.means_ #Getting means of data
clusters_plot(K,X, GMM_prediction, means_train) #Plotting clusters
print('The purity score of test data by GMM clustering (K =',K,'):',purity_score(true_train,GMM_prediction))
GMM_prediction_test=gmm.predict(test_data) #Predicting the test data
clusters_plot(K,Y, GMM_prediction_test, means_train) #Plotting clusters of test data
print('The purity score of test data by GMM clustering (K =',K,'):',purity_score(true_test,GMM_prediction_test)) 

print("-------------------------------Question 3-----------------------------\n")
#DBSCAN
dbscan_model=DBSCAN(eps=5, min_samples=10).fit(train_data) #Preparing DBSCAN model and fitting data
DBSCAN_predictions = dbscan_model.labels_ #Getting labels
plt.scatter(X[:,0], X[:,1], c=DBSCAN_predictions,cmap='rainbow', label ='Cluster') #Plotting scatter plot
plt.title('Clusters of eps=5 and min_samples = 10')
plt.xlabel('Dimention-1')
plt.ylabel('Dimension-2')
plt.show()
print('The purity score of the train data by DBSCAN clustering (eps =5,min = 10):',purity_score(true_train,DBSCAN_predictions))
DBSCAN_prediction_test=dbscan_model.fit_predict(test_data) #Predicting the test data
k2=plt.scatter(Y[:,0], Y[:, 1], c=DBSCAN_prediction_test,cmap='rainbow' , label ='Cluster 1') #Plotting test data
plt.colorbar(k2)
plt.title('Clusters when eps = 5 qnd min_sample = 10')
plt.xlabel('Dimention-1')
plt.ylabel('Dimension-2')
plt.show()
print('The purity score of the test data by DBSCAN clustering ((eps =5,min = 10)',purity_score(true_test,DBSCAN_prediction_test)) 

print("-------------------------------Bonus Question 1-----------------------------\n")
k=[ 2, 5, 8, 12, 18 ,20]
dis=[]
for i in k:
    kmean1 = KMeans(n_clusters = i,random_state=42) #Preparing model for different k values
    kmean1.fit(train_data)  #Fitting training data
    dis.append(sum(np.min(cdist(X, kmean1.cluster_centers_,'euclidean'),axis=1)) / X.shape[0]) #Finding distortion measure
    kmeans_prediction1=kmean1.predict(train_data)
    print('The purity score of the train data by KMeans clustering (K =',i,'):',purity_score(true_train,kmeans_prediction1))
#Plotting elbow graph 
plt.plot(k,dis)
plt.grid()
plt.scatter(k,dis,c='orange')
plt.xlabel('k-values')
plt.ylabel('Distrion measure')
plt.show()

kmean3 = KMeans(n_clusters = 8,random_state=42) #Preparing model for best k value
kmean3.fit(train_data) #Fitting training data
prediction3=kmean3.predict(train_data) #Predicting the data
clusters_plot(8,X,prediction3,centers) #Plotting data

list2=[]
for i in k:
    gmm5 = GaussianMixture(n_components = i,random_state=42) #Making GMM model
    gmm5.fit(train_data) #Fitting training the data
    p2=gmm5.score_samples(train_data) #Finding the log likelihood
    prediction5=gmm5.predict(train_data) #Predicting purity score
    list2.append(p2.sum())
    print('The purity score of the train data by GMM clustering (K =',i,'):',purity_score(true_train,prediction5))
plt.plot(k,list2) #Plotting elbow graph
plt.grid()
plt.scatter(k,list2,c='orange')
plt.xlabel('k values')
plt.ylabel('Total log likelihood')
plt.show()

gmm3= GaussianMixture(n_components = 8,random_state=42) #Preparing GMM model
gmm3.fit(train_data) #Fitting training data
GMM_prediction3 = gmm3.predict(train_data) #Predicting training data
means_train3=gmm3.means_ #Finding means of data
clusters_plot(8,X,GMM_prediction3,means_train3) #Plot the best fit graph

print("-------------------------------Bonus Question 2-----------------------------\n")
elist=[1,5,10]
mlist=[1,10,30,50]
for i in elist:
    for j in mlist:
        dbscan_model1=DBSCAN(eps=i, min_samples=j).fit(train_data)  #DBSCAN model for the different values of eps and min_samples
        DBSCAN_predictions1 = dbscan_model1.labels_ #Finding labels
        k5=plt.scatter(X[:,0], X[:,1], c=DBSCAN_predictions1,cmap='rainbow') #Plotting scatter plot
        plt.colorbar(k5)
        plt.title('DBSCAN plot when eps='+str(i)+' min_samples ='+str(j))
        plt.xlabel('Dimention-1')
        plt.ylabel('Dimension-2')
        plt.show()
        print('The purity score of the train data by DBSCAN clustering ((eps =',i,',min_samples =',j,')',purity_score(true_train,DBSCAN_predictions1))
        
