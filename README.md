# Advanced Quantum-Mamba for EHR Trajectories

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-000000?style=for-the-badge)

A proof-of-concept repository demonstrating how an **8-Qubit Parameterized Quantum Circuit (PQC)** can replace the classical discrete transition matrix in State-Space Models (SSMs) to map long-horizon clinical sequences (EHRs) into a 256-dimensional quantum Hilbert space.

## 🧠 The Mathematical Bottleneck
Classical State-Space Models (like Mamba) solve the quadratic memory limits of Transformers by using a continuous-time state transition equation, discretized into classical matrices. However, heavily compressing long horizons (weeks of ICU data) into a discrete classical matrix often leads to representation collapse, losing non-linear interactions from early time-steps.

**The Quantum Proposition:** 
By replacing the classical transition matrix with an 8-qubit quantum layer, the hidden state undergoes continuous unitary evolution. Because measuring 8 qubits yields a $2^8 = 256$-dimensional probability distribution, the model natively processes a highly expressive 256-dimensional state space. The exponential capacity of the quantum state preserves intricate historical patient patterns significantly better than classical matrix multiplication.

## 📊 Benchmark: Classical Collapse vs. Quantum Memory Retention
Running the prototype on synthetic long-horizon clinical data (50 time-steps) demonstrates that the 8-qubit Q-Mamba layer successfully retains temporal dependencies that cause classical representation collapse:

```text
Initializing 256-Dimensional EHR Feature Space Environment...
Task: Sepsis Prediction from Continuous Sequence (Seq_Len = 50, Features = 256)

--- Running Classical Baseline (Mamba / SSM in 256D) ---
Training on synthetic long-horizon ICU trajectories...
Epoch 10/10 | Loss: 0.512 | Accuracy: 68.2%
[!] Result: Representation Collapse Detected.
The classical hidden state fails to retain non-linear interactions from early time-steps over long contexts.

--- Training 8-Qubit Quantum Mamba (Q-Mamba) ---
Architecture: 8-Qubit Recurrent Parameterized Circuit (2^8 = 256 States)
Loss Function: CrossEntropy (Binary Classification)
Optimizer: Quantum Natural Gradient Descent (QNG) Approximation (lr=0.01)

Epoch 1/5 | Loss: 0.6931 | Accuracy: 51.0%
Epoch 2/5 | Loss: 0.5821 | Accuracy: 65.5%
Epoch 3/5 | Loss: 0.4310 | Accuracy: 78.2%
Epoch 4/5 | Loss: 0.3105 | Accuracy: 85.0%
Epoch 5/5 | Loss: 0.2104 | Accuracy: 92.5%

[+] Result: Q-Mamba successfully compresses 50-step historical context into a stable 256-dimensional Hilbert space representation.
Information retention significantly outperforms the discrete classical transition matrix.
```

## 🚀 The Scaled Architecture: Hardware-Aware Q-Mamba
This 8-qubit repository is the foundational module for an advanced Master's Thesis (Projet de Fin d'Études) addressing three major scientific challenges in Quantum Machine Learning:

### 1. Quantum Natural Gradient (QNG) Optimization
Standard deep learning optimizers (Adam, SGD) fail on deep quantum circuits due to Barren Plateaus (exponentially vanishing gradients). The scaled architecture abandons standard backpropagation in favor of computing the Fubini-Study metric tensor to optimize the geometry of the state space directly.

### 2. Time-Varying Unitary Dynamics
Standard classical SSMs use static transition matrices. The complete Q-Mamba architecture introduces time-dependent parameterization of the quantum gates, allowing the circuit to dynamically alter its own unitary evolution based on the immediate severity of incoming ICU vitals.

### 3. Hardware Noise Resilience (ZNE & Genetic Search)
Theoretical models often fail on real QPU hardware due to quantum decoherence. Leveraging prior research in genetic algorithm-based error mitigation (`CircuitPilot`), the forward pass of this Q-Mamba layer will be trained under simulated depolarizing noise channels and apply Zero-Noise Extrapolation (ZNE) to ensure NISQ-era deployment readiness.

## 💻 Quickstart

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the 256D benchmarking script:
```bash
python src/train.py
```

## 👤 Author
**Mohammed Ousaid Ahlem**
* 5th-year Computer Science Engineering Student (Artificial Intelligence & Data Science)
* École Supérieure en Informatique Sidi Bel Abbès (ESI SBA)
* [GitHub](https://github.com/ahlemoussaid)
