import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

def prepare_and_train_model(learning_rate, max_epochs, goal):
    data = {
        'Math': [80, 90, 50, 60, 70],
        'Science': [85, 95, 55, 65, 75],
        'English': [78, 88, 58, 68, 72],
        'Pass': [1, 1, 0, 0, 1]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('student_scores.csv', index=False)
    
    df = pd.read_csv('student_scores.csv')
    X = df[['Math', 'Science', 'English']].values
    y = df['Pass'].values

    global mean_X, std_X
    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0)
    X = (X - mean_X) / std_X

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    perceptron = Perceptron(num_inputs=X.shape[1], learning_rate=learning_rate)
    perceptron.fit(X_train, y_train, max_epochs)

    np.savez('perceptron_model.npz', weights=perceptron.weights)

    y_pred = perceptron.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return perceptron, accuracy, X_train, y_train 

def predict():
    try:
        math = float(entry_math.get())
        science = float(entry_science.get())
        english = float(entry_english.get())
        data = np.array([[math, science, english]])
        data = (data - mean_X) / std_X
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
        global perceptron, X, y
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
root.configure(background='#2596be')
root.geometry("500x500")

# Centering widgets
frame = tk.Frame(root, bg='#2596be')
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

def apply_shadow(widget):
    widget.config(highlightbackground="black", highlightthickness=2)

labels = {
    "Learning Rate:": (0, 0),
    "Max Epochs:": (1, 0),
    "Goal (Accuracy):": (2, 0),
    "Math:": (4, 0),
    "Science:": (5, 0),
    "English:": (6, 0)
}

entries = {
    "learning_rate": (0, 1),
    "max_epochs": (1, 1),
    "goal": (2, 1),
    "math": (4, 1),
    "science": (5, 1),
    "english": (6, 1)
}

for text, grid in labels.items():
    label = tk.Label(frame, text=text, bg='#2596be', fg='white', font=('Arial', 12, 'bold'))
    label.grid(row=grid[0], column=grid[1], padx=10, pady=10)

entry_learning_rate = tk.Entry(frame, font=('Arial', 12))
entry_max_epochs = tk.Entry(frame, font=('Arial', 12))
entry_goal = tk.Entry(frame, font=('Arial', 12))
entry_math = tk.Entry(frame, font=('Arial', 12))
entry_science = tk.Entry(frame, font=('Arial', 12))
entry_english = tk.Entry(frame, font=('Arial', 12))

entries_widgets = [
    entry_learning_rate,
    entry_max_epochs,
    entry_goal,
    entry_math,
    entry_science,
    entry_english
]

for widget, grid in zip(entries_widgets, entries.values()):
    widget.grid(row=grid[0], column=grid[1], padx=10, pady=10)
    apply_shadow(widget)

train_button = tk.Button(frame, text='Train Model', command=train_model, font=('Arial', 12, 'bold'))
train_button.grid(row=3, column=1, pady=20)
apply_shadow(train_button)

predict_button = tk.Button(frame, text='Predict', command=predict, font=('Arial', 12, 'bold'))
predict_button.grid(row=7, column=1, pady=20)
apply_shadow(predict_button)

root.mainloop()
