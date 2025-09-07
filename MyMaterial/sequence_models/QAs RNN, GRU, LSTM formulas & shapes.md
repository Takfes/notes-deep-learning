# 1) Vanilla RNN

### Forward

* **Activation** :
    $$
    a^{\langle t\rangle} = g\!\left(W_{aa}a^{\langle t-1\rangle} + W_{ax}x^{\langle t\rangle} + b_a\right)
    $$

* **Output** :
    $$
    \quad
    y^{\langle t\rangle}=g_{\text{out}}\!\left(W_{ya}a^{\langle t\rangle}+b_y\right)
    $$

### Forward (compact form)

Starting from :

$$
W_{aa}a^{\langle t-1\rangle}+W_{ax}x^{\langle t\rangle}+b_a.
$$

Form the block matrix and stacked input:

* Concatenated weight matrix $W_{\alpha}$ :
$$
W_a=\;\begin{bmatrix}W_{aa}&W_{ax}\end{bmatrix}\in\mathbb{R}^{h\times(h+d)}
$$

* Concatenated activation matrix $x^{\langle t\rangle}$ :
$$
\quad
\bar{x}^{\langle t\rangle}=\begin{bmatrix}a^{\langle t-1\rangle}\\ x^{\langle t\rangle}\end{bmatrix}\in\mathbb{R}^{h+d}.
$$

* Resulting compact form :
$$ W_a\,\bar{x}^{\langle t\rangle}=W_{aa}a^{\langle t-1\rangle}+W_{ax}x^{\langle t\rangle}$$
* Apply $g(\cdot)$ and add $b_a$ as before.

### Shapes

* Input dim $d$, hidden dim $h$, output dim $o$.
* **Split:** $W_{ax}\in\mathbb{R}^{h\times d},\;W_{aa}\in\mathbb{R}^{h\times h},\;b_a\in\mathbb{R}^{h\times1},\;W_{ya}\in\mathbb{R}^{o\times h},\;b_y\in\mathbb{R}^{o\times1}$.
* **Concatenated:** $W_a\in\mathbb{R}^{h\times(h+d)}$ (just $[W_{aa}\;W_{ax}]$ side-by-side), $b_a\in\mathbb{R}^{h\times1}$

---

# 2) GRU

### Equations

* **Update gate** $\Gamma_u$:

  $$
  \Gamma_u^{\langle t\rangle}=\sigma\!\left(W_u\,[a^{\langle t-1\rangle},\,x^{\langle t\rangle}] + b_u\right)
  $$
* **Reset gate** $\Gamma_r$:

  $$
  \Gamma_r^{\langle t\rangle}=\sigma\!\left(W_r\,[a^{\langle t-1\rangle},\,x^{\langle t\rangle}] + b_r\right)
  $$
* **Candidate** $\tilde{c}^{\langle t\rangle}$ (a.k.a. candidate hidden):

  $$
  \tilde{h}^{\langle t\rangle}=\tilde{c}^{\langle t\rangle}=\tanh\!\left(W_c\,[\Gamma_r^{\langle t\rangle}\!\odot a^{\langle t-1\rangle},\,x^{\langle t\rangle}] + b_c\right)
  $$
* **Hidden / output** $a^{\langle t\rangle}$:

  $$
  h^{\langle t\rangle}=a^{\langle t\rangle}=c^{\langle t\rangle}=(1-\Gamma_u^{\langle t\rangle})\odot a^{\langle t-1\rangle}+\Gamma_u^{\langle t\rangle}\odot \tilde{c}^{\langle t\rangle}
  $$

### Shapes

* Input dim $d$, hidden dim $h$, output dim $o$.
* Gates / states: $\Gamma_u^{\langle t\rangle},\Gamma_r^{\langle t\rangle},\tilde{c}^{\langle t\rangle},a^{\langle t\rangle},c^{\langle t\rangle}\in\mathbb{R}^{h\times1}$.
* **Concatenated weights:** $W_u,W_r,W_c\in\mathbb{R}^{h\times(h+d)}$, biases $b_u,b_r,b_c\in\mathbb{R}^{h\times1}$.
* **Split shape (if used):** $ W\in\mathbb{R}^{h\times d}, U\in\mathbb{R}^{h\times h}$

---

# 3) LSTM

### Equations

* **Forget gate** $\Gamma_f$:

  $$
  \Gamma_f^{\langle t\rangle}=\sigma\!\left(W_f\,[a^{\langle t-1\rangle},\,x^{\langle t\rangle}] + b_f\right)
  $$
* **Input gate** $\Gamma_u$ (a.k.a. “update”/“input” gate):

  $$
  \Gamma_u^{\langle t\rangle}=\sigma\!\left(W_u\,[a^{\langle t-1\rangle},\,x^{\langle t\rangle}] + b_u\right)
  $$
* **Candidate** $\tilde{c}^{\langle t\rangle}$:

  $$
  \tilde{c}^{\langle t\rangle}=\tanh\!\left(W_c\,[a^{\langle t-1\rangle},\,x^{\langle t\rangle}] + b_c\right)
  $$
* **Cell update**:

  $$
  c^{\langle t\rangle}=\Gamma_f^{\langle t\rangle}\odot c^{\langle t-1\rangle}+\Gamma_u^{\langle t\rangle}\odot \tilde{c}^{\langle t\rangle}
  $$
* **Output gate** $\Gamma_o$:

  $$
  \Gamma_o^{\langle t\rangle}=\sigma\!\left(W_o\,[a^{\langle t-1\rangle},\,x^{\langle t\rangle}] + b_o\right)
  $$
* **Hidden/output**:

  $$
  a^{\langle t\rangle}=\Gamma_o^{\langle t\rangle}\odot \tanh\!\left(c^{\langle t\rangle}\right)
  $$

### Shapes

* Input dim $d$, hidden dim $h$, output dim $o$.
* Gates / states: $\Gamma_f^{\langle t\rangle},\Gamma_u^{\langle t\rangle},\Gamma_o^{\langle t\rangle},\tilde{c}^{\langle t\rangle},c^{\langle t\rangle},a^{\langle t\rangle}\in\mathbb{R}^{h\times1}$.
* **Concatenated weights:** $W_f,W_u,W_o,W_c\in\mathbb{R}^{h\times(h+d)}$, biases $b_f,b_u,b_o,b_c\in\mathbb{R}^{h\times1}$.
* **Split shape (if used):** for each gate $g\in\{f,u,o,c\}$, $W_{g x}\in\mathbb{R}^{h\times d},\;U_{g a}\in\mathbb{R}^{h\times h}$