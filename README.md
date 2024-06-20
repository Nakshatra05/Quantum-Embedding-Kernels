# Quantum Embedding Kernels for Binary Classification

This repository implements a quantum machine learning model using Quantum Embedding Kernels (QEKs) for a binary classification task. The implementation leverages PennyLane for quantum circuit simulation and Scikit-Learn for classical SVM classification.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/qek_classifier.git
    cd qek_classifier
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the script:
    ```bash
    python qek_classifier.py
    ```

2. The script will output the classification accuracy of the SVM classifier on the test set.

## Files

- `qek_classifier.py`: Main script containing the implementation of the quantum feature map, QEK matrix computation, optimization, and SVM classifier training.
- `requirements.txt`: List of dependencies required to run the script.

## References

- [Quantum Embedding Kernels for Machine Learning](https://arxiv.org/pdf/2105.02276)
- [PennyLane Documentation](https://pennylane.ai/qml/demos/tutorial_kernels_module/)

