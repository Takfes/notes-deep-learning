# Forward and Backward Propagation Formulas

## Forward Propagation

### 1. Compute $ Z $
$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$

### 2. Compute $ A $

$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$

---

## Backward Propagation

### 1. Compute $ dZ $

$$
dZ^{[l]} = dA^{[l]} \cdot g^{[l]\prime}(Z^{[l]})
$$

### 2. Compute $ dW $

$$
dW^{[l]} = \frac{1}{m} * dZ^{[l]} A^{[l-1]T}
$$

### 3. Compute $ db $

$$
db^{[l]} = \frac{1}{m} * np.sum(dz^{[l]}, axis = 1, keepdims = True)
$$

### 4. Compute $ dA^{[l-1]} $

$$
da^{[l-1]} = W^{[l]T} dz^{[l]}
$$

---

## Compute Error

### 1. Classifications - Binary Cross Entropy Loss

$$
\mathcal{L}_{\text{BCE}} = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

### 2. Classifications - Cross Entropy Loss (Multi-class)

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

### 3. Regression - Mean Squared Error (MSE)

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### 4. Regression - Mean Absolute Error (MAE)

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$


---

## Activation Functions

### Sigmoid

$$
g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Derivative:**
$$
\sigma'(z) = \sigma(z) \left(1 - \sigma(z)\right)
$$

---

### Tanh

$$
g(z) = \tanh(z)
$$

**Derivative:**
$$
\tanh'(z) = 1 - \tanh^2(z)
$$

---

### ReLU

$$
g(z) = \max(0, z)
$$

**Derivative:**
$$
g'(z) =
\begin{cases}
1 & \text{if } z > 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

### Leaky ReLU

$$
g(z) =
\begin{cases}
z & \text{if } z > 0 \\
\alpha z & \text{otherwise}
\end{cases}
$$

**Derivative:**
$$
g'(z) =
\begin{cases}
1 & \text{if } z > 0 \\
\alpha & \text{otherwise}
\end{cases}
$$