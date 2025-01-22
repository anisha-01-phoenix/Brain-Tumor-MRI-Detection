# scripts/evaluate.py
from tensorflow.keras.models import load_model
from load_data import load_train_test_data

data_dir = '../data'  
model_dir = '../models/model.h5'

# Load the model
print(f"Loading the model from {model_dir}...")
model = load_model(model_dir)

# Load the test data
print(f"Loading test data from {data_dir}...")
_, X_test, _, y_test, _ = load_train_test_data(data_dir)

# Evaluate the model on the test set
print(f"Evaluating the model on the test data...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')
