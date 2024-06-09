import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import messagebox

# Data preparation and model training
def prepare_and_train_model():
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

    # Initialize and train the perceptron
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)

    # Test the model
    y_pred = perceptron.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    # Save the model
    joblib.dump(perceptron, 'perceptron_model.pkl')

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

# Run model preparation and training
prepare_and_train_model()

# Load the trained perceptron model
perceptron = joblib.load('perceptron_model.pkl')

# GUI setup
root = tk.Tk()
root.title("Pass/Fail Predictor")

tk.Label(root, text="Math:").grid(row=0)
tk.Label(root, text="Science:").grid(row=1)
tk.Label(root, text="English:").grid(row=2)

entry_math = tk.Entry(root)
entry_science = tk.Entry(root)
entry_english = tk.Entry(root)

entry_math.grid(row=0, column=1)
entry_science.grid(row=1, column=1)
entry_english.grid(row=2, column=1)

tk.Button(root, text='Predict', command=predict).grid(row=3, column=1, pady=4)

root.mainloop()
