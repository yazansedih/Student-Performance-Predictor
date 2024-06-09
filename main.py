import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import messagebox

# Data preparation and model training
def prepare_and_train_model(learning_rate, max_epochs, goal):
    """
    Prepares the dataset, trains the Perceptron model with specified hyperparameters, and saves it to a file.
    """
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
    X = df[['Math', 'Science', 'English']]
    y = df['Pass']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Perceptron model with specified hyperparameters
    perceptron = Perceptron(eta0=learning_rate, max_iter=max_epochs)

    # Train the Perceptron model using the training set
    perceptron.fit(X_train, y_train)

    # Predict the outcomes for the test set
    y_pred = perceptron.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Save the trained model to a file
    joblib.dump(perceptron, 'perceptron_model.pkl')

    # Return the trained model
    return perceptron

# GUI prediction
def predict():
    try:
        math = float(entry_math.get())
        science = float(entry_science.get())
        english = float(entry_english.get())
        data = [[math, science, english]]
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
        global perceptron  # Add this line to access the global variable
        perceptron = prepare_and_train_model(learning_rate, max_epochs, goal)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy >= goal:
            messagebox.showinfo("Training Result", f'Model trained successfully with accuracy: {accuracy}')
        else:
            messagebox.showwarning("Training Result", f'Model accuracy ({accuracy}) did not meet the goal ({goal}).')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numerical values for training parameters.")

# Load the trained perceptron model (initialize as None)
perceptron = None

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
