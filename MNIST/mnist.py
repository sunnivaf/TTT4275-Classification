import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import cluster
import os
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
import seaborn as sn

def load_mnist_data(num_train, num_test):
    with open('MNist_ttt4275/train_images.bin','rb') as binFile:
        train_images = binFile.read()
    
    with open('MNist_ttt4275/train_labels.bin','rb') as binFile:
        train_labels = binFile.read()
    
    with open('MNist_ttt4275/test_images.bin','rb') as binFile:
        test_images = binFile.read()
    
    with open('MNist_ttt4275/test_labels.bin','rb') as binFile:
        test_labels = binFile.read()
    
    train_images = np.reshape(np.frombuffer(train_images[16:16+784*num_train], dtype=np.uint8), (num_train,784))
    train_images = train_images.astype(dtype=np.float16)/255 
    train_labels = np.frombuffer(train_labels[8:num_train+8], dtype=np.uint8)
    test_images = np.reshape(np.frombuffer(test_images[16:16+784*num_test], dtype=np.uint8), (num_test,784))
    test_images = test_images.astype(dtype=np.float16)/255
    test_labels = np.frombuffer(test_labels[8:num_test+8], dtype=np.uint8)

    return train_images, train_labels, test_images, test_labels


def plot_confusion_matrix(title, cm):
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cm.flatten()/np.sum(cm)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
            zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(10,10)
    ax = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label');
    ax.xaxis.set_ticklabels(['0','1','2','3','4','5','6','7','8','9'])
    ax.yaxis.set_ticklabels(['0','1','2','3','4','5','6','7','8','9'])
    plt.show()
    return

def error_rate(cm):
    errors = 0
    num_samples = 0
    for true_label in range(len(cm)):
        for pred_label in range(len(cm)):
            if true_label != pred_label:
                errors += cm[true_label][pred_label]
            num_samples += cm[true_label][pred_label]
    return errors/num_samples

def majority_vote(K,distances,train_labels):
    indexes_sorted_by_distance = np.argsort(np.array(distances),axis=0)
    class_votes = [0]*10
    class_distance = [0]*10

    for k in range(K):
        class_votes[int(train_labels[indexes_sorted_by_distance[k]])] += 1
        class_distance[int(train_labels[indexes_sorted_by_distance[k]])] += distances[indexes_sorted_by_distance[k]]

    #Selecting the candidate class with the highest number of votes
    cand_class = class_votes.index(max(class_votes))
    cand_count = class_votes[cand_class]
    cand_total_distance = class_distance[cand_class]

    #Tie in votes
    for i in range(10):
        count = class_votes[i]
        total_distance = class_distance[i]

        if ((count == cand_count) and (total_distance < cand_total_distance)):
            cand_class = i

    return cand_class

def KNN_classifier(K, train_images, train_labels, test_images):
    start = time.time()
    predictions = []

    for i in range(len(test_images)): 
        distances = []
        for j in range(len(train_images)):
            euclid_dist = distance.euclidean(test_images[i], train_images[j])
            distances.append(euclid_dist)
        majority_class = majority_vote(K,distances,train_labels)
        predictions.append(majority_class)
    
    end = time.time()
    processing_time = end-start
    return predictions, processing_time

def NN_classifier(train_images, train_labels, test_images):
    start = time.time()
    predictions = []

    for i in range(len(test_images)): 
        min = float('inf')
        for j in range(len(train_images)):
            euclid_dist = distance.euclidean(test_images[i], train_images[j])
            if euclid_dist < min:
                min = euclid_dist
                prediction = train_labels[j]
        predictions.append(prediction)
        
    end = time.time()
    processing_time = end-start
    return predictions, processing_time 


def clustering(M, data_images, data_labels):
    N = 10
    templates_by_class = []
    for n in range(N):
        t = []
        for i,label in enumerate(data_labels):
            if n == label:
                t.append(data_images[i])
        templates_by_class.append(t)

    clusters = []
    labels = []
    for i in range(N):
        part = templates_by_class[i]
        kmeans = cluster.KMeans(n_clusters=M).fit(part)
        clusters.append(kmeans.cluster_centers_)
        labels.append([i]*M)

    cluster_images = np.array(clusters).flatten().reshape((M*N,data_images.shape[1]))
    cluster_labels = np.array(labels,dtype=int).flatten().reshape(M*N,1)
    return cluster_images, cluster_labels

def show_classification(predictions, test_labels, test_images, misclassified, num_digits):
    num = 0
    i = len(predictions)
    while num < num_digits:
        i -= 1
        if misclassified == 'true':
            if predictions[i] != test_labels[i]:
                plt.imshow(test_images[i].reshape(28,28))
                plt.title(f"Classified {test_labels[i]} as {predictions[i]}")
                plt.show()
                num += 1
        if misclassified == 'false':
            if predictions[i] == test_labels[i]:
                plt.imshow(test_images[i].reshape(28,28))
                plt.title(f"Correctly classified {test_labels[i]}")
                plt.show()
                num += 1
    return


def main():
    num_train = 60000
    num_test = 10000
    K = 7
    M = 64

    train_images, train_labels, test_images, test_labels = load_mnist_data(num_train,num_test)

    #TASK 1

    predictions_nn, processingtime_nn = NN_classifier(train_images, train_labels, test_images)

    cm_nn = confusion_matrix(test_labels, predictions_nn)
    plot_confusion_matrix("Confusion matrix - 1NN classifier without clustering", cm_nn)
    error_rate_nn = error_rate(cm_nn)

    print(f"processingtime for NN: {processingtime_nn/60} minutes")
    print(f"Error rate without clustering: {error_rate_nn}")
    
    #Plot some of the misclassified and correctly classified pixtures:
    misclassified = 'true'
    show_classification(predictions_nn, test_labels, test_images, misclassified, 20)
    misclassified = 'false'
    show_classification(predictions_nn, test_labels, test_images, misclassified, 20)

    #TASK 2

    cluster_images, cluster_labels = clustering(M, train_images, train_labels)
    predictions_cnn, processingtime_c = NN_classifier(cluster_images, cluster_labels, test_images)

    cm_cnn = confusion_matrix(test_labels, predictions_cnn)
    plot_confusion_matrix("Confusion matrix - 1NN classifier with clustering", cm_cnn)
    error_rate_cnn = error_rate(cm_cnn)

    print(f"processingtime with clustering: {processingtime_c/60} minutes")
    print(f"Error rate with clustering: {error_rate_cnn}")

    #TASK 2c
    predictions_knn, processingtime_knn = KNN_classifier(K, cluster_images, cluster_labels, test_images)

    cm_knn = confusion_matrix(test_labels, predictions_knn)
    plot_confusion_matrix("Confusion matrix - 7NN classifier with clustering", cm_knn)
    error_rate_knn = error_rate(cm_knn)

    print(f"processingtime for KNN: {processingtime_knn/60} minutes")
    print(f"Error rate for KNN: {error_rate_knn}")
   
    plt.show()
    return

main()
