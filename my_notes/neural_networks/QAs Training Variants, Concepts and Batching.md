# Training Variants, Concepts and Batching

## Key Concepts

- **Epoch**: One complete forward and backward pass of *all* training samples in the dataset.
- **Batch Size**: Number of training samples processed in one forward and backward pass.
- **Number of Iterations**: Number of gradient update steps per epoch, each step using `batch_size` samples.

- **Example:**  
Dataset size = 100 samples, Batch size = 20 → 100 / 20 = **5 iterations per epoch**

**Note:**  
Each iteration involves:
- Computing the loss function
- Performing backpropagation
- Updating model weights

---

## Training Variants Overview

- **Batch Gradient Descent**  
    - `batch_size =` entire dataset  
    - 1 iteration per epoch

- **Mini-batch Gradient Descent**  
    - `1 < batch_size <` dataset size  
    - Most common approach

- **Stochastic Gradient Descent (SGD)**  
    - `batch_size = 1`  
    - One sample per iteration


## Batching Strategies: How the Gradient is Computed

- **Batch Gradient Descent (BGD):**
    - Uses the entire training set for each gradient calculation.
    - Results in a smooth, direct optimization path with monotonically decreasing cost.
    - Very slow per iteration for large datasets.
    - No stochastic effects.

- **Stochastic Gradient Descent (SGD) (Pure):**
    - Uses one single training example per gradient calculation.
    - Leads to an extremely noisy optimization path that oscillates around the minimum and may not converge exactly.
    - Inefficient due to loss of vectorization speed-up.
    - Highest stochasticity.

- **Mini-Batch Gradient Descent:**
    - Uses a subset (mini-batch) of training examples (e.g., 64-512).
    - The most common approach, balancing efficiency (vectorization) and rapid progress by taking many steps per epoch.
    - Cost function generally trends downwards but is "a little bit noisier" due to moderate stochastic effects.

## Optimizers vs. Batching Strategy

- **Batching Strategy:**  
    Determines how many examples are used to compute the gradient for a single update step. This directly influences the "stochastic effects" or noise level.

- **Optimizers:**  
    Algorithms (e.g., SGD, Adam, RMSProp, Momentum) that define how weights are updated based on a calculated gradient (or an exponentially weighted average of gradients/squared gradients). They are separate mechanisms for the weight update rule. Frameworks typically provide general "gradient descent optimizers" or "Adam optimizers".

## SGD Optimizer Naming Convention

- The SGD optimizer in deep learning frameworks (like PyTorch) is a general gradient descent algorithm for updating weights. It implements the update rule, but does not inherently enforce a batch size of 1.
- Its "stochastic" nature is determined by your chosen batching strategy when you feed data to it.
- Therefore, an SGD optimizer would perform Batch Gradient Descent (BGD) if your data pipeline is configured to supply gradients computed over the entire dataset for each update. The naming is largely a practical convention, as mini-batch gradient descent (which introduces stochasticity) is the most common application in deep learning.