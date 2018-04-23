import cv2
import numpy as np
import os
import tools as tl

from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extractingData(dirName, label):
    images = []
    imagesSmall = []
    labels = []
    for root, dirs, files in os.walk(dirName):
        for file in files:
            imagepath = os.path.join(root, file)
            if file.endswith('.jpg'):
                img = cv2.imread(imagepath, 0)
                img = cv2.resize(img, (100, 20))
                th_adap = cv2.adaptiveThreshold(img, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
                images.append(th_adap)
                labels.append(label)

    return images, labels

if __name__ == '__main__':

    pos_images, pos_labels = extractingData("dataset/pos", 1)
    neg_images, neg_labels = extractingData("dataset/neg", 0)
    images = pos_images + neg_images
    labels = pos_labels + neg_labels

    '''
        Hog
    '''

    cell = 4
    pw = 100
    ph = 20
    nbin = 4
    
    featureVector = (pw / cell) * (ph / cell) * nbin

    hog = cv2.HOGDescriptor(_winSize=(cell, cell),
                                _blockSize=(cell, cell),
                                _blockStride=(cell, cell),
                                _cellSize=(cell, cell),
                                _nbins=nbin, _histogramNormType = 0, _gammaCorrection = True)

    
    
    features = []
    for img in images:
        features.append(hog.compute(img).reshape(featureVector))


    x = np.asarray(features)
    y = np.asarray(labels)

    print x.shape
    print y.shape

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    clf = linear_model.LogisticRegression(C=1e4)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print accuracy_score(y_test, y_pred)
