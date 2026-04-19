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

* Sparsity loss = **L1 penalty on gate values**, encouraging pruning

---

## 🧠 Key Idea

The model **automatically identifies and removes less important connections** during training.

* Low λ → less pruning, higher accuracy
* High λ → more pruning, lower accuracy

---

## 📊 Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1e-3   | 0.30     | 9%       |
| 3e-3   | 0.34     | 10%      |
| 5e-3   | 0.32     | 10%      |

---

## 📈 Observations

* Increasing λ slightly increases sparsity
* Accuracy remains stable across λ values
* Model successfully prunes a portion of weights during training

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

This project demonstrates that **neural networks can dynamically adapt their structure** using learnable gates and L1 regularization.

It shows how pruning can be integrated directly into training, making models more efficient for real-world deployment.

---

## 📎 Reference

Based on Tredence AI Engineering Internship case study.


<img width="465" height="92" alt="image" src="https://github.com/user-attachments/assets/5a5d76af-086f-4bd5-acc8-743e7434c601" />
