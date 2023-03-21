import math
import pickle
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn import svm

# Implementing PCA from scratch
class Principal_Component_Analysis:

    # constructor
    def __init__(self, n, exclude_indices=[]):
        self.n = n                                      # The number of principal components to keep
        self.exclude_indices = exclude_indices          # Drop these principal components
        self.mean_data = None                           # The average face
        self.weights = None                             # The weights of each face
        self.eigenvalues = None                         # The eigenvalues of the covariance matrix
        self.principal_components = None                # The unit eigenvectors of the covariance matrix
    
    # fit the PCA to the data
    def fit(self, data):

        # Find the average face
        self.mean_data = np.mean(data, axis=0)

        # Subtract all the faces with the average face to center it
        data_adj = data - self.mean_data

        # Calculate the covariance matrix
        # Computing A*A.T speeds up calculation compared to A.T*A [transpose trick]
        C = 1/(len(data_adj) - 1) * np.matmul(data_adj, data_adj.T)

        # Find the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(C)

        # Sort them in descending order
        sorted_indices = np.argsort(eigenvalues.T)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]

        # Recovering the eigenfaces
        new_eigenvectors = np.matmul(data_adj.T, eigenvectors).T[sorted_indices]

        # Normalizing the eigenfaces
        self.principal_components = new_eigenvectors / np.linalg.norm(new_eigenvectors)

        # Save the weights
        self.weights = np.matmul(data_adj, self.principal_components.T)
    
    # transform the data according to the fitted model
    def transform(self, data):
        # center the data
        data_adj = data - self.mean_data

        components = self.principal_components[0:self.n]

        bool_arr = np.ones(len(components), dtype=bool)
        bool_arr[self.exclude_indices] = False
        weights = np.matmul(data_adj, components[bool_arr].T)

        # return the weights that make up the face
        return weights
    
    # perform fitting and transforming at the same time
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    # show the mean face
    def show_mean(self, resolution):
        plt.title("Mean Image")
        plt.imshow(self.mean_data.reshape(*resolution), cmap="gray")
        plt.show()

    # show the cumulative explained variance
    def show_explained_variance(self):
        total = sum(self.eigenvalues)
        cumulative_sum = [sum(self.eigenvalues[:i]) / total for i in range(self.n)]
        plt.plot(range(1, self.n + 1), cumulative_sum)
        plt.title("Explained Variance")
        plt.xlabel("Number of Principle Components")
        plt.show()

    # show the eigenfaces
    def show_components(self, resolution):
        plot_dim = math.floor(math.sqrt(self.n))
        fig, axes = plt.subplots(plot_dim,plot_dim,sharex=True,sharey=True,figsize=(8,10))
        fig.suptitle("Principle Components")
        for i in range(plot_dim ** 2):
            ax = axes[i // plot_dim][i % plot_dim]
            ax.axis("off")
            ax.imshow(self.principal_components[i].reshape(*resolution), cmap="gray")
        plt.show()
    
    # show all the metrics at the same time
    def show_metrics(self, resolution):
        self.show_mean(resolution=resolution)
        self.show_explained_variance()
        self.show_components(resolution=resolution)

class Standard_Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, matrix):
        transposed = np.array(matrix).T
        self.mean = np.expand_dims(transposed.mean(axis=1), axis=1)
        self.std = np.expand_dims(transposed.std(axis=1), axis=1)
        self.std[self.std == 0] = 1
    def transform(self, matrix):
        return ((np.array(matrix).T - self.mean) / self.std).T
    def fit_transform(self, matrix):
        self.fit(matrix)
        return self.transform(matrix)


# Classifier to determine the most similar face that a sample belongs to based on the PCA weights based on Euclidean distance
class Euclidean_Distance_Classifier:
    # constructor
    def __init__(self):
        self.X = []     # the face vectors
        self.y = []     # the labels for those face vectors (i.e. the names of the people)
    
    # train the model
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    # give the model a single face and get back a single prediction
    # if only_prediction is False, then more information will be returned as a dictionary (the index, and the distance)
    def predict_single(self, data, only_prediction=True):
        # find the distances of the sample from the training faces
        distances = np.linalg.norm(data - self.X, axis=1)

        # find the index of the smallest distance
        index = np.argmin(distances)

        # output the prediction based on the labels of the training faces and return a result
        prediction = self.y[index]
        if only_prediction:
            return prediction
        else:
            return {"prediction": prediction, "index": index, "distance": distances[index]}
    
    # predict a batch of faces at the same time, returns an array of predictions
    def predict(self, data):
        predictions = [self.predict_single(sample) for sample in data]
        return predictions
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return find_accuracy(y_pred, y_test)

# Classifier to predict the class that the sample belows to based on its k nearest neighbors
class K_Nearest_Neighbors(Euclidean_Distance_Classifier):
        def __init__(self, neighbors=3, *args):
            super().__init__(*args)
            self.neighbors = neighbors

        def predict_single(self, data, only_prediction=True):
            # find the distances of the sample from the training faces
            distances = np.linalg.norm(data - self.X, axis=1)

            # find the index of the smallest distance
            sorted = np.argsort(distances)
            neighbors = np.array(self.y)[sorted][:self.neighbors]
            index = np.argmin(distances)

            # output the prediction based on the labels of the training faces and return a result
            prediction = mode(neighbors)
            if only_prediction:
                return prediction
            else:
                return {"prediction": prediction, "index": index, "distance": distances[index]}
            
# Save data as a pickle file
def save_data(data, path):
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

# Load data from a pickle file
def load_data(path):
    pickle_in = open(path,"rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data

# find the accuracy of model based on its predictions (pred) and the actual values (true)
# outputs a value from 0 (totally inaccurate) to 1 (perfectly accurate)
def find_accuracy(pred, true):
    accuracy = sum([1 if pred_face == true_face else 0 for pred_face, true_face in zip(pred, true)]) / len(pred)
    return accuracy

def evaluate_models(X, y, resolution=(100, 100), n_components=50, exclude_indices=[], is_scaler=True, is_show_metrics=True, classifier="euclidean"):

    if classifier not in ["euclidean", "knn", "linearsvc"]:
        print("Invalid classifier. Choose from euclidean, knn and svc.")
        return
    
    # Evaluate the model based on k-fold cross validation
    kfold = StratifiedKFold(n_splits=5)
    predictions, scores = {"pred":[], "true":[]}, []

    for train, test in kfold.split(X, y):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        
        # Scale the data so that it has zero mean and unit variance
        if is_scaler:
            scaler = Standard_Scaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Conduct PCA on the data
        pca = Principal_Component_Analysis(n=n_components, exclude_indices=exclude_indices)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Train and evaluate classifier model
        if classifier == "euclidean":
            model = Euclidean_Distance_Classifier()
        elif classifier == "knn":
            model = K_Nearest_Neighbors(neighbors=5)
        elif classifier == "linearsvc":
            model = svm.LinearSVC(dual=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(find_accuracy(y_pred, y_test))
        predictions["pred"].extend(y_pred)
        predictions["true"].extend(y_test)
    
    # Print the average 5-fold validation scores
    accuracy = round(sum(scores)/len(scores) * 100, 5)
    print(f"Average accuracy for {classifier} classifier: ", accuracy, "%")

    # Show the confusion matrices for the classifiers
    if is_show_metrics:
        pca.show_metrics(resolution=resolution)
        ax = plt.axes()
        plt.suptitle(f"Confusion matrix for PCA / {classifier}")
        plt.title(f"Accuracy: {accuracy}%", fontsize=10)
        ConfusionMatrixDisplay.from_predictions(
            predictions["true"], 
            predictions["pred"], 
            xticks_rotation="vertical",
            include_values=False,
            normalize="true",
            display_labels=["" for i in range(len(set(predictions["true"])))],
            ax=ax)
        ax.tick_params(axis='both', which='both', length=0)
        plt.show()
