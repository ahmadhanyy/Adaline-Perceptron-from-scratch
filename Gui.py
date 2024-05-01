import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from Models import Models
from Models import Data

def on_select(event):
    selected_value = feature_1.get()

x_train = None
y_train = None
x = None
y = None
weights = None
feature1 = None
feature2 = None
class1 = None
class2 = None

def get_text():
    global x_train, y_train, x, y, weights, feature1, feature2, class1, class2
    feature1 = feature_1.get()
    feature2 = feature_2.get()
    class1 = class_1.get()
    class2 = class_2.get()
    if feature1 == feature2:
        messagebox.showinfo("Message", "Selected features are the same.")
    if class1 == class2:
        messagebox.showinfo("Message", "Selected Classes are the same.")
    NumberOfEpochs = int(epochs.get())
    MSEThreshold = float(threshold.get())
    ETALearningRate = float(learnrate.get())
    data = Data()
    x_train, x_test, y_train, y_test, x, y = data.data_preprocessing(class1=class1, class2=class2, feature1=feature1, feature2=feature2)

    if bias.get() == 1:
        bias_value = True
    else:
        bias_value = False
    selected_model = var.get()
    model = Models(ETALearningRate, NumberOfEpochs)
    if selected_model == 1:
        weights = model.train_adaline(x_train, y_train, bias=bias_value, threshold=MSEThreshold)
        accuracy, confusion_matrix = model.test(x_test, y_test, weights, selected_model)
    else:
        weights = model.train_perceptron(x_train, y_train, bias=bias_value)
        accuracy, confusion_matrix = model.test(x_test, y_test, weights, selected_model)
    # Display the accuracy and the confusion matrix in the GUI window
    accuracy_label.config(text=f"Accuracy: {accuracy*100}%")
    confusion_matrix_label.config(text=f"Confusion Matrix: \n{confusion_matrix}")
def show_graph():
    global x_train, y_train, x, y, weights, feature1, feature2, class1, class2
    # Generating the decision boundary
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Flattening the mesh grid and predicting the labels
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    Z = np.dot(mesh_data, weights)
    Z = np.where(Z >= 0, 1, -1)
    Z = Z.reshape(xx.shape)

    # Plotting the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Scatter the data points for class 1 (label 1)
    plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], label=class1, marker='o', color='blue')

    # Scatter the data points for class -1 (label -1)
    plt.scatter(x_train[y_train == -1][:, 0], x_train[y_train == -1][:, 1], label=class2, marker='x', color='red')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.title('Decision Boundary and Data Points')
    plt.show()

m = tk.Tk()
var = tk.IntVar()
m.title("First Task")
m.geometry("500x500")

# Create and place labels
tk.Label(m, text='First Class').grid(row=0, pady=10)
tk.Label(m, text='Second Class').grid(row=1, pady=10)
tk.Label(m, text='First Feature').grid(row=2, pady=10)
tk.Label(m, text='Second Feature').grid(row=3, pady=10)
tk.Label(m, text='Number of Epochs').grid(row=4, pady=10)
tk.Label(m, text='MSE Threshold').grid(row=5, pady=10)
tk.Label(m, text='ETA Learning rate').grid(row=6, pady=10)

# Create and place radio buttons
R1 = tk.Radiobutton(m, text="Adaline Algorithm", variable=var, value=1)
R2 = tk.Radiobutton(m, text="Perceptron Algorithm", variable=var, value=2)
R1.grid(row=7, column=0, padx=10)
R2.grid(row=7, column=1, padx=10)

# Create and place check buttons
bias = tk.IntVar()
c1 = tk.Checkbutton(m, text="Add Bias", variable=bias, onvalue=1, offvalue=0)
c1.grid(row=7, column=3, padx=10)

# Create and place entry fields
class_1 = tk.StringVar()
combo3 = ttk.Combobox(m, textvariable=class_1)
combo3['values'] = ('BOMBAY', 'CALI', 'SIRA')
combo3.bind('<<ComboboxSelected>>', on_select)
combo3.grid(row=0, column=1, pady=10)

class_2 = tk.StringVar()
combo4 = ttk.Combobox(m, textvariable=class_2)
combo4['values'] = ('BOMBAY', 'CALI', 'SIRA')
combo4.bind('<<ComboboxSelected>>', on_select)
combo4.grid(row=1, column=1, pady=10)

feature_1 = tk.StringVar()
combo = ttk.Combobox(m, textvariable=feature_1)
combo['values'] = ('Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes')
combo.bind('<<ComboboxSelected>>', on_select)
combo.grid(row=2, column=1, pady=10)

feature_2 = tk.StringVar()
combo2 = ttk.Combobox(m, textvariable=feature_2)
combo2['values'] = ('Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes')
combo2.bind('<<ComboboxSelected>>', on_select)
combo2.grid(row=3, column=1, pady=10)

epochs = tk.Entry(m)
epochs.grid(row=4, column=1, pady=10)
threshold = tk.Entry(m)
threshold.grid(row=5, column=1, pady=10)
learnrate = tk.Entry(m)
learnrate.grid(row=6, column=1, pady=10)

# Create and place the execute button
execute_button = tk.Button(m, text="Execute", width=15, command=get_text)
execute_button.grid(row=9, column=0, pady=20)
graph_button = tk.Button(m, text="Show Graph", width=15, command=show_graph)
graph_button.grid(row=9, column=1, pady=10)

# Label widgets to display accuracy and confusion matrix
accuracy_label = tk.Label(m, text="Accuracy: ")
accuracy_label.grid(row=11, column=0, pady=10)

confusion_matrix_label = tk.Label(m, text="Confusion Matrix: ")
confusion_matrix_label.grid(row=11, column=1, pady=10)

m.mainloop()
