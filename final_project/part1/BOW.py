import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.cluster import KMeans
import cv2
import os
from sklearn import svm
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import heapq


def load_images(dir):
    train = {}
    for filename in os.listdir(dir):
        category = []
        path = dir + "/" + filename
        for cato in os.listdir(path):
            img = cv2.imread(path + "/" + cato, 0)
            if img is not None:
                category.append(img)
        train[filename] = category
    return train


def load_images_for_final(dir):
    train = []
    for filename in os.listdir(dir):
        path = dir + "/" + filename
        for cato in os.listdir(path):
            img = plt.imread(path + "/" + cato)
            if img is not None:
                train.append(img)
    return train


def extract_feature(train, number_of_subset, method):
    feature_vector = []
    des_list = []
    ylist = []
    if method == 'SIFT_gray':
        sift = cv2.xfeatures2d.SIFT_create()
        for key, images in train.items():
            for img in images[:number_of_subset]:
                kp, des = sift.detectAndCompute(img, None)
                des_list.extend(des)
            for img1 in images[number_of_subset:]:
                kp, des = sift.detectAndCompute(img1, None)
                feature_vector.append(des)
                ylist.append(key)

    elif method == 'hog':
        for key, images in train.items():
            for img in images[:number_of_subset]:
                kp, des = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), block_norm='L2-Hys',
                              visualize=True)
                des_list.extend(des)
            for img1 in images[number_of_subset:]:
                kp, des = hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), block_norm='L2-Hys',
                              visualize=True)
                feature_vector.append(des)
                ylist.append(key)

    elif method == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        for key, images in train.items():
            for img in images[:number_of_subset]:
                kp, des = surf.detectAndCompute(img, None)
                des_list.extend(des)
            for img1 in images[number_of_subset:]:
                kp, des = surf.detectAndCompute(img1, None)
                feature_vector.append(des)
                ylist.append(key)

    elif method == 'SIFT_RGB':
        sift = cv2.xfeatures2d.SIFT_create()
        for key, images in train.items():
            for img in images[:number_of_subset]:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                kp, des = sift.detectAndCompute(gray_img, None)
                _, desR = sift.compute(img[..., 0], keypoints=kp)
                _, desG = sift.compute(img[..., 1], keypoints=kp)
                _, desB = sift.compute(img[..., 2], keypoints=kp)
                des = np.hstack((desR, desG, desB))
                des_list.extend(des)
            for img1 in images[number_of_subset:]:
                gray_img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                kp, des = sift.detectAndCompute(gray_img, None)
                _, desR = sift.compute(img1[..., 0], keypoints=kp)
                _, desG = sift.compute(img1[..., 1], keypoints=kp)
                _, desB = sift.compute(img1[..., 2], keypoints=kp)
                des = np.hstack((desR, desG, desB))
                feature_vector.append(des)
                ylist.append(key)

    return feature_vector, ylist, des_list


def visual_vocabulary(cluster_number, dis_list):
    km = KMeans(n_clusters=cluster_number, n_init=10)
    km.fit(dis_list)
    visual_vocab = km.cluster_centers_
    return visual_vocab



def econding(feature_vector, vocabulary, method, cluster_number=400):
    im_features = np.zeros((len(feature_vector), cluster_number), "float32")
    if method == 'SIFT_RGB':
        for i in range(len(feature_vector)):
            if feature_vector[i][0] is not None:
                words, distance = vq(feature_vector[i], vocabulary)
                for w in words:
                    im_features[i][w] += 1
            else:
                continue
    else:
        for i in range(len(feature_vector)):
            if feature_vector[i] is not None:
                words, distance = vq(feature_vector[i], vocabulary)
                for w in words:
                    im_features[i][w] += 1
            else:
                continue

    return im_features


def normalize(dict_feature):
    stdSlr = StandardScaler().fit(dict_feature)
    dict_feature = stdSlr.transform(dict_feature)
    return dict_feature


def change_label(train_y, label):
    train_y = np.array(train_y)
    label_change = []
    for i in train_y:
        if i == label:
            label_change.append(1)
        else:
            label_change.append(0)
    return label_change


def SVM_fit(dict_feature, train_y):
    ##normalized
    clf = svm.SVC()
    clf.fit(dict_feature, np.array(train_y))
    return clf


def predict(clf, feature, y_true):
    pred = (clf.predict(feature))
    accuracy = accuracy_score(y_true, pred)
    yscore = clf.decision_function(feature)
    MAP = average_precision_score(y_true, yscore)
    print('Accuracy is: ', accuracy)
    print('MAP is: ', MAP)


def rank_image(clf, feature):
    maxlist = []
    minlist = []
    test = load_images_for_final('./test')
    yscore = clf.decision_function(feature)
    yscore = list(yscore)
    max_num_index_list = map(yscore.index, heapq.nlargest(5, yscore))
    min_num_index_list = map(yscore.index, heapq.nsmallest(5, yscore))
    for index, i in enumerate(list(max_num_index_list)):
        maxlist.append(test[i])
    for index, i in enumerate(list(min_num_index_list)):
        minlist.append(test[i])
    return maxlist, minlist


def plot_histogram():
    feature = np.load('./result/pure_feature.npy')
    plt.style.use("seaborn-darkgrid")

    def hisss(feature, label):
        if label == 'car':
            k = np.arange(0, 300)
        elif label == 'airplane':
            k = np.arange(300, 600)
        elif label == 'bird':
            k = np.arange(600, 900)
        elif label == 'ship':
            k = np.arange(900, 1200)
        else:
            k = np.arange(1200, 1500)
        car = 0
        for i in k:
            car += feature[i] / sum(feature[i])
        car = car / sum(car)
        plt.bar(x=np.arange(0, 400), height=car, color='red')
        plt.title(label, size=20)
        plt.ylim(0, 0.015)

    plt.figure(figsize=(20, 10))
    label = ['car', 'airplane', 'bird', 'ship', 'horse']
    for index, i in enumerate(label):
        plt.subplot(2, 3, index + 1)
        hisss(feature, i)
    #     plt.savefig('./plot/hist.jpg', dpi=400, bbox_inches='tight')
    plt.show()


def plot_rankimages():
    plt.figure(figsize=(20, 20))
    max_400 = np.load('./result/siftgray400_max.npy')
    min_400 = np.load('./result/siftgray400_min.npy')
    for index, i in enumerate(min_400):
        plt.subplot(5, 5, index + 1)
        plt.imshow(i)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    for index, i in enumerate(max_400):
        plt.subplot(5, 5, index + 1)
        plt.imshow(i)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_keypoints():
    train = load_images('./train')
    sift = cv2.xfeatures2d.SIFT_create()
    for key, value in train.items():
        kp = sift.detect(value[0], None)
        img = cv2.drawKeypoints(value[0], kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img)
        plt.show()


def main_function(clusters, Method):
    ###loading images
    print('-----loading images')
    train = load_images('./train')

    ###generate subset for voca and rest for classifier
    ## rest_feature is rest images features
    ### train_y is rest images features label
    ### des_list_for_voca is used to generate voca
    print('-----extracting features')
    rest_feature_vector, train_y, des_list_for_voca = extract_feature(train, number_of_subset=200, method=Method)
    plot_keypoints()
    ### generate visual vocabulary and normalized.
    print('-----vocabulary')
    #     visual = visual_vocabulary(cluster_number=clusters,dis_list=des_list_for_voca)
    #     feature = econding(rest_feature_vector, method = Method, visual,cluster_number=clusters)
    #     feature = normalize(feature)

    ##### I already saved all voca and dic we can directly use this. we can ingore previous step
    visual = np.load('./result/visual_400.npy')
    feature = np.load('./result/dict_feature_400.npy')
    plot_histogram()

    #### train model
    print('-----Training model')
    # let target label become 1 and others become 0 ##
    test = load_images('./test')
    test_x, test_y, _ = extract_feature(test, number_of_subset=0, method=Method)
    test_feature = econding(test_x, visual, method=Method, cluster_number=clusters)
    test_feature = normalize(test_feature)

    label = ['car', 'airplane', 'bird', 'ship', 'horse']
    totalmax = []
    totalmin = []
    for i in label:
        print('This is result of' + i)
        target_label = change_label(train_y, i)
        ##svm
        clf = SVM_fit(feature, target_label)

        ### process test
        print('-----predict')
        target_test_label = change_label(test_y, i)
        predict(clf, test_feature, target_test_label)
        maxlist, minlist = rank_image(clf, test_feature)
        totalmax.extend(maxlist)
        totalmin.extend(minlist)
    plot_rankimages()

    return totalmax, totalmin


totalmax, totalmin = main_function(400, Method='SIFT_gray')