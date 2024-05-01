import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Data:

    def data_preprocessing(self, class1, class2, feature1, feature2):
        # Load the dataset from the Excel file
        file_path = 'Dry_Bean_Dataset.csv'
        df = pd.read_csv(file_path)
        
        # Assuming you have a DataFrame 'df' with a 'Class' column
        # Map the two chosen classes
        df['Class'] = df['Class'].map({class1: 1, class2: -1})

        # Fill NAN values
        df.interpolate(method='linear', inplace=True)
        #df = df.fillna(df.mean())

        # The first 5 columns are the features
        features = df.iloc[:, :5]  # Extracting the first 5 columns as features

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit and transform the scaler on the features
        normalized_features = scaler.fit_transform(features)

        # Replace the original columns with the normalized ones
        df.iloc[:, :5] = normalized_features

        # Separate rows with Class values of -1 and 1 after mapping
        class1_df = df[df['Class'] == 1]
        class2_df = df[df['Class'] == -1]

        # Set a random seed for reproducibility
        np.random.seed(42)

        # Shuffle each class rows
        class1_df = class1_df.sample(frac=1, random_state=42).reset_index(drop=True)
        class2_df = class2_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Extract features and target separately for each class
        X1 = class1_df[[feature1, feature2]].values
        X2 = class2_df[[feature1, feature2]].values
        y1 = class1_df['Class'].values
        y2 = class2_df['Class'].values

        X1_train = X1[:30]
        X2_train = X2[:30]
        X1_test = X1[30:50]
        X2_test = X2[30:50]
        X_train = np.concatenate((X1_train, X2_train), axis=0)
        X_test = np.concatenate((X1_test, X2_test), axis=0)
        x = np.concatenate((X_train, X_test), axis=0)

        y1_train = y1[:30]
        y2_train = y2[:30]
        y1_test = y1[30:50]
        y2_test = y2[30:50]
        y_train = np.concatenate((y1_train, y2_train), axis=0)
        y_test = np.concatenate((y1_test, y2_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        return X_train, X_test, y_train, y_test, x, y

class Models:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def train_adaline(self, x, y, bias, threshold):
        weights = np.random.rand(x.shape[1])
        if bias:
            biasValue = np.random.rand()
        else:
            biasValue = 0

        for _ in range(self.n_iterations):
            errors = []
            pred = []
            for Xi, Yi in zip(x, y):
                predict = np.dot(Xi, weights.T) + biasValue
                error = Yi - predict
                weights = weights + (self.learning_rate * error * Xi)
                if biasValue != 0:
                    biasValue = biasValue + (self.learning_rate * error)
                errors.append(error)
                pred.append(predict)
            mse = 0
            for Pi, Yi in zip(pred, y):
                mse += (Yi - Pi) ** 2
            mse = mse / y.size
            if mse < threshold:
                break
        return weights, biasValue

    def train_perceptron(self, x, y, bias):
        weights = np.random.rand(x.shape[1])
        if bias:
            biasValue = np.random.rand()
        else:
            biasValue = 0

        for _ in range(self.n_iterations):
            for Xi, Yi in zip(x, y):
                predict = np.dot(Xi, weights.T) + biasValue
                if predict >= 0:
                    predict = 1
                else:
                    predict = -1
                if predict != Yi:
                    error = Yi - predict
                    weights = weights + (self.learning_rate * error * Xi)
                    if biasValue != 0:
                        biasValue = biasValue + (self.learning_rate * error)
            return weights, biasValue

    def test(self, x, y, weights, biasValue, modelnumber):
        predictions = np.dot(x, weights) + biasValue
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1
        correct = np.sum(predictions == y)
        accuracy = correct / len(y) * 100
        if modelnumber == 1:
            print(f"Accuracy of the Adaline model on the test data: {accuracy:.2f}%")
        else:
            print(f"Accuracy of the Perceptron model on the test data: {accuracy:.2f}%")

        confusion_matrix = np.zeros((2, 2))
        for i in range(len(y)):
            if y[i] == 1:
                if predictions[i] == 1:
                    confusion_matrix[0, 0] += 1  # True Positive
                else:
                    confusion_matrix[0, 1] += 1  # False Negative
            else:
                if predictions[i] == 1:
                    confusion_matrix[1, 0] += 1  # False Positive
                else:
                    confusion_matrix[1, 1] += 1  # True Negative

        print("Confusion Matrix:")
        print(confusion_matrix)
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        return accuracy, confusion_matrix

#data = Data()
#X_train, X_test, y_train, y_test, x, y = data.data_preprocessing('SIRA', 'CALI', 'Area', 'Perimeter')

#model = Models(learning_rate=0.01, n_iterations=100)
# Train the Adaline model
#weights, biasValue = model.train_adaline(X_train, y_train, bias=False, threshold=0.2)
# Train the Perceptron model
#weights, biasValue = model.train_perceptron(X_train, y_train, bias=True)

# Testing the models (for Adaline: modelnumber=1) (for Perceptron: modelnumber=2)
#model.test(X_test, y_test, weights, biasValue, modelnumber=1)
