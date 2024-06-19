import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

from perceptron import Perceptron

def prepare_and_train_model(learning_rate, max_epochs, goal):
    # Sample dataset
    data = {
        'Math': [80, 90, 50, 60, 70],
        'Science': [85, 95, 55, 65, 75],
        'English': [78, 88, 58, 68, 72],
        'Pass': [1, 1, 0, 0, 1]
    }

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('student_scores.csv', index=False)

    # Load dataset
    df = pd.read_csv('student_scores.csv')
    X = df[['Math', 'Science', 'English']].values
    y = df['Pass'].values

    # Normalize the data
    global mean_X, std_X  # Declare as global to use in prediction
    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0)
    X = (X - mean_X) / std_X

    # Initialize the Perceptron model with specified hyperparameters
    perceptron = Perceptron(num_inputs=X.shape[1], learning_rate=learning_rate)

    # Train the Perceptron model
    perceptron.fit(X, y, max_epochs)

    # Save the trained model to a file (simplified for demonstration purposes)
    np.savez('perceptron_model.npz', weights=perceptron.weights)

    # Calculate final accuracy
    y_pred = perceptron.predict(X)
    accuracy = np.mean(y_pred == y)
    return perceptron, accuracy, X, y

def predict():
    try:
        math = float(entry_math.get())
        science = float(entry_science.get())
        english = float(entry_english.get())
        data = np.array([[math, science, english]])
        data = (data - mean_X) / std_X  # Normalize input
        prediction = perceptron.predict(data)
        result = 'Pass' if prediction[0] == 1 else 'Fail'
        messagebox.showinfo("Result", f'The prediction is: {result}')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numerical values.")

def train_model():
    try:
        learning_rate = float(entry_learning_rate.get())
        max_epochs = int(entry_max_epochs.get())
        goal = float(entry_goal.get())
        global perceptron, X, y  # Add this line to access the global variables
        perceptron, accuracy, X, y = prepare_and_train_model(learning_rate, max_epochs, goal)
        if accuracy >= goal:
            messagebox.showinfo("Training Result", f'Model trained successfully with accuracy: {accuracy}')
        else:
            messagebox.showwarning("Training Result", f'Model accuracy ({accuracy}) did not meet the goal ({goal}).')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numerical values for training parameters.")

# Load the trained perceptron model (initialize as None)
perceptron = None
mean_X = None
std_X = None

# GUI setup
root = tk.Tk()
root.title("Pass/Fail Predictor")

# Training parameters
tk.Label(root, text="Learning Rate:").grid(row=0)
tk.Label(root, text="Max Epochs:").grid(row=1)
tk.Label(root, text="Goal (Accuracy):").grid(row=2)

entry_learning_rate = tk.Entry(root)
entry_max_epochs = tk.Entry(root)
entry_goal = tk.Entry(root)

entry_learning_rate.grid(row=0, column=1)
entry_max_epochs.grid(row=1, column=1)
entry_goal.grid(row=2, column=1)

tk.Button(root, text='Train Model', command=train_model).grid(row=3, column=1, pady=4)

# Prediction inputs
tk.Label(root, text="Math:").grid(row=4)
tk.Label(root, text="Science:").grid(row=5)
tk.Label(root, text="English:").grid(row=6)

entry_math = tk.Entry(root)
entry_science = tk.Entry(root)
entry_english = tk.Entry(root)

entry_math.grid(row=4, column=1)
entry_science.grid(row=5, column=1)
entry_english.grid(row=6, column=1)

tk.Button(root, text='Predict', command=predict).grid(row=7, column=1, pady=4)

root.mainloop()