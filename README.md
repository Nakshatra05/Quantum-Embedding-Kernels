# Quantum Embedding Kernels for Binary Classification

This repository implements a quantum machine learning model using Quantum Embedding Kernels (QEKs) for a binary classification task. The implementation leverages PennyLane for quantum circuit simulation and Scikit-Learn for classical SVM classification.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Nakshatra05/Quantum-Embedding-Kernels.git
    cd qek_classifier
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the script**:
    ```bash
    python qek_classifier.py
    ```

2. **Output**:
    The script will output the classification accuracy of the SVM classifier on the test set. For example:
    ```
    Classification accuracy: 0.85
    ```

## Files

- `qek_classifier.py`: Main script containing the implementation of the quantum feature map, QEK matrix computation, optimization, and SVM classifier training.
- `requirements.txt`: List of dependencies required to run the script.

## Explanation

### Quantum Feature Map
The quantum feature map is a quantum circuit that embeds data points into quantum states. In this implementation, single-qubit rotations are used to encode a 2-dimensional data point into a quantum state.

### Quantum Embedding Kernel (QEK)
The QEK is computed by evaluating the overlap between the quantum states produced by the quantum feature map for different data points. The kernel function is defined as:
\[ k(x, x') = |\langle \phi(x') | \phi(x) \rangle|^2 \]
This kernel measures the similarity between data points in the quantum feature space.

### Training and Evaluation
After computing the QEK matrix for the training and test datasets, an SVM classifier is trained using the training QEK matrix. The classification accuracy of the SVM classifier is then evaluated on the test set.

## References

- [Quantum Embedding Kernels for Machine Learning](https://arxiv.org/pdf/2105.02276)
- [PennyLane Documentation](https://pennylane.ai/qml/demos/tutorial_kernels_module/)

## License

This project is licensed under the MIT License.
