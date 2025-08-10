"""
This script demonstrates the equivalence between two approaches for computing the pre-activation values in a simple recurrent neural network (RNN) cell:
1. Separately multiplying the hidden state and input vectors by their respective weight matrices and summing the results.
2. Concatenating the weight matrices and input vectors, then performing a single matrix multiplication.

By showing that both methods yield the same result, the script illustrates how merging weight matrices and input vectors can lead to more efficient computation in deep neural networks, particularly in the context of RNNs.
"""

import numpy as np

# -------------------------------
# Define matrices
# -------------------------------
waa = np.array([[1, 2], [3, 4]])
a = np.array([[1], [2]])

wax = np.array([[1, 2, 1, -1], [-1, 1, 1, 2]])
x = np.array([[1], [2], [3], [4]])

# -------------------------------
# Calculation w/o concatenation
# -------------------------------
# R11
waa.shape
a.shape
R11 = waa @ a
R11.shape

# R12
wax.shape
x.shape
R12 = wax @ x
R12.shape

# R1
R1 = R11 + R12
R1.shape

# -------------------------------
# Calculation with concatenation
# -------------------------------
# Concatenate waa and wax horizontally
W = np.hstack((waa, wax))
W.shape

# Concatenate a and x vertically
AX = np.vstack((a, x))
AX.shape

# R2
R2 = W @ AX
R2.shape

# -------------------------------
# Check if R1 and R2 are equal
# -------------------------------
R1 == R2
