import numpy as np


class KNearestNeighbors:

    def __init__(self, distances, labels, k=10):
        self.distances = distances
        self.labels = labels
        self.k = k

    def _kNN(self, instance, train, k):
        nearest = np.argpartition(self.distances[instance][train], k)
        nearest_labels = self.labels[train][nearest[:k]]
        unique, counts = np.unique(nearest_labels, return_counts=True)
        counts = dict(zip(unique, counts))
        
        probabilities = np.zeros(10)
        for i in range(10):
            probabilities[i] = 0 if i not in counts else counts[i] / k
            
        return probabilities

    def score(self, train, test):
        correct = 0
        total = 0
        confusion = np.zeros((10,10))
        # choose k to be at most as large as supported by the training dataset
        # or as configured, if enough training samples are available
        k = min(len(train)//10, self.k)
        for i in test:
            probs = self._kNN(i, train, k)
            pred = np.argmax(probs)
            confusion[self.labels[i]][pred] += 1
            if pred == self.labels[i]:
                correct += 1
            total += 1
        accuracy = correct / total
        return accuracy, confusion

class KNearestNeighborsTrainTest(KNearestNeighbors):
    
    def __init__(self, distances, train_labels, test_labels, k=10):
        self.test_labels = test_labels
        super().__init__(distances, train_labels, k)

    def score(self, train):
        correct = 0
        total = 0
        confusion = np.zeros((10,10))
        # choose k to be at most as large as supported by the training dataset
        # or as configured, if enough training samples are available
        k = min(len(train)//10, self.k)
        for i, label in enumerate(self.test_labels):
            probs = self._kNN(i, train, k)
            pred = np.argmax(probs)
            confusion[label][pred] += 1
            if pred == label:
                correct += 1
            total += 1
        accuracy = correct / total
        return accuracy, confusion
        
