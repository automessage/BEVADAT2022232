import numpy as np
from sklearn.cluster import KMeans

from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class KMeansOnDigits:
    def __init__(self, n_clusters:int, random_state:int):
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def load_dataset(self):
        self.digits = load_digits()
    
    def predict(self):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = self.model.fit_predict(self.digits.data, self.digits.target)
    
    def get_labels(self):
        self.labels = np.empty(shape=self.clusters.shape)

        for i in self.digits.target_names:
            mask = self.clusters == i
            self.labels[mask] = mode(self.digits.target[mask], keepdims=False).mode
    
    def calc_accuracy(self):
        self.accuracy = np.round(accuracy_score(self.digits.target, self.labels), 2)
    
    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)