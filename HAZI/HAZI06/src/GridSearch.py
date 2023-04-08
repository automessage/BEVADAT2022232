from src.Loader import Loader
from src.DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class DecesionTreeGridSearch:

    def __init__(self, possible_split_values = range(2, 7), possible_depth_values = range(3, 10)):
        self.possible_split_values = possible_split_values
        self.possible_depth_values = possible_depth_values
        self.decision_tree = DecisionTreeClassifier(min_samples_split=5)

    def searchBestDepth(self, X_train, X_test, Y_train, Y_test):
        results = []
        max_result = (0, 0, 0)
        
        for min_samples_split in self.possible_split_values:
            self.decision_tree.min_samples_split = min_samples_split

            for max_depth in self.possible_depth_values:
                loading = Loader('Checking depth - ' + str(min_samples_split) + ", " + str(max_depth))
                loading.start()
                self.decision_tree.root = None
                self.decision_tree.max_depth = max_depth
                try:
                    self.decision_tree.fit(X_train, Y_train)
                    Y_pred = self.decision_tree.predict(X_test)
                    results.append((min_samples_split, max_depth, accuracy_score(Y_test, Y_pred)))
                    if (max_result[2] < results[-1][2]):
                        max_result = results[-1]
                except:
                    results.append((min_samples_split, max_depth, '**error**'))

                loading.terminate()
        return max_result, results

