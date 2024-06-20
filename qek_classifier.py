import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the quantum feature map
def quantum_feature_map(x, wires):
    qml.RX(x[0], wires=wires[0])
    qml.RY(x[1], wires=wires[1])

# Compute QEK matrix
def compute_qek_matrix(data, wires):
    num_data = len(data)
    qek_matrix = np.zeros((num_data, num_data))

    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def kernel(x1, x2):
        quantum_feature_map(x1, wires)
        qml.adjoint(quantum_feature_map)(x2, wires)
        return qml.probs(wires=[0])
    
    for i in range(num_data):
        for j in range(num_data):
            qek_matrix[i, j] = np.abs(kernel(data[i], data[j]))**2
    
    return qek_matrix

# Generate synthetic dataset
def generate_dataset():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])
    return X, y

# Main function
def main():
    # Generate dataset
    X, y = generate_dataset()

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define wires for quantum circuit
    wires = [0, 1]

    # Compute QEK matrix for training and testing data
    K_train = compute_qek_matrix(X_train, wires)
    K_test = compute_qek_matrix(X_test, wires)

    # Train SVM classifier using the QEK matrix
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)

    # Evaluate the classifier
    y_pred = svm.predict(K_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classification accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()

