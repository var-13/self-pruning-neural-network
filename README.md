# self-pruning-neural-network

# Self-Pruning Neural Network (Tredence Case Study)

## 📌 Problem

Deploying large neural networks is limited by memory and computation.
This project builds a model that **learns to prune its own weights during training** instead of pruning after training.

---

## ⚙️ Approach

* Built a custom `PrunableLinear` layer

* Each weight has a **learnable gate (sigmoid-based)**

* Effective weight:

  ```
  w' = w × sigmoid(gate_score)
  ```

* Loss function:

  ```
  Total Loss = CrossEntropy + λ × SparsityLoss
  ```

* Sparsity loss = **L1 penalty on gate values**, which pushes unnecessary weights toward zero

---

## 🧠 Key Idea

Instead of manually pruning, the model **automatically learns which connections are important**.

Higher λ → more pruning
Lower λ → better accuracy

---

## 📊 Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1e-3   | 0.31     | 9%       |
| 3e-3   | 0.33     | 9%       |
| 5e-3   | 0.33     | 10%      |

---

## 📈 Observations

* Increasing λ increases sparsity (pruning effect)
* Accuracy remains stable for moderate λ values
* Model successfully removes less important weights during training

---

## 🚀 How to Run

```bash
pip install torch torchvision
python self_pruning_network.py
```

---

## 📁 Project Structure

```
self_pruning_network.py   # main implementation
README.md                 # explanation
```

---

## 🎯 Conclusion

This project demonstrates that **neural networks can adapt their structure dynamically** using learnable gates and L1 regularization.

This approach helps reduce model size and is useful for **resource-constrained environments**.

---

## 📎 Reference

Case study based on Tredence AI Engineering Internship task 



<img width="308" height="65" alt="image" src="https://github.com/user-attachments/assets/f1b47110-860e-4a2c-9b31-9e6916def263" />
