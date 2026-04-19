# Self-Pruning Neural Network (Tredence Case Study)

## 📌 Problem

Deploying large neural networks is limited by memory and computation.
This project builds a model that **learns to prune its own weights during training**, instead of applying pruning after training.

---

## ⚙️ Approach

* Implemented a custom `PrunableLinear` layer

* Each weight is controlled by a **learnable gate (sigmoid function)**

* Effective weight:

  ```
  w' = w × sigmoid(gate_score)
  ```

* Loss function:

  ```
  Total Loss = CrossEntropy + λ × SparsityLoss
  ```

* Sparsity loss = **L1 penalty on gate values**, which encourages unnecessary weights to move toward zero

---

## 🧠 Key Idea

The model **automatically identifies and removes less important connections** during training.

* Low λ → less pruning, better accuracy
* High λ → more pruning, potential accuracy drop

---

## 📊 Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1e-3   | 0.30     | 9%       |
| 3e-3   | 0.34     | 10%      |
| 5e-3   | 0.32     | 10%      |

---

## 📈 Observations

* Increasing λ increases sparsity, showing a trade-off with accuracy
* Accuracy remains relatively stable for moderate λ values
* The model successfully prunes a portion of weights during training
* **λ = 3e-3 provides the best balance between accuracy and sparsity**

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

This project demonstrates that neural networks can **adapt their structure dynamically** using learnable gates and L1 regularization.

It shows how pruning can be integrated directly into the training process, making models more efficient for **resource-constrained environments**.

---

## 📎 Reference

Based on Tredence AI Engineering Internship case study.

<img width="266" height="52" alt="image" src="https://github.com/user-attachments/assets/13f4da26-f396-47f1-8fa5-fcdcbaf520dc" />
