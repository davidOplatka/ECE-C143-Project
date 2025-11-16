## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

pip install -e .

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. Train model: `python ./scripts/train_model.py`


---

## Transformer Tunable Hyperparams

The Transformer-based decoder exposes several hyperparameters that control model capacity, sequence resolution, regularization, and positional encoding. Below is a concise guide to all configurable parameters and how they affect performance.

---

## **1. Core Transformer Architecture**

These parameters define the size and depth of the Transformer encoder.

### **`hidden_dim` (a.k.a. d_model)**

* **What it is:** Embedding dimension used by the Transformer.
* **Typical values:** 256, 512, 768, 1024
* **Notes:** Must be divisible by `nhead`.

### **`layer_dim`**

* **What it is:** Number of Transformer encoder layers.
* **Typical values:** 2–6
* **Effect:** Higher = deeper, more expressive, slower.

### **`nhead`**

* **What it is:** Number of attention heads.
* **Typical values:**

  * `hidden_dim=256 → nhead=4 or 8`
  * `hidden_dim=512 → nhead=8`
  * `hidden_dim=1024 → nhead=8 or 16`
* **Notes:** `hidden_dim % nhead == 0` required.

### **`dim_feedforward`**

* **What it is:** Size of the feedforward network inside each Transformer layer.
* **Typical values:** `2–4 × hidden_dim` (e.g., 1024–4096)
* **Effect:** Larger FFN adds expressive power.

---

## **2. Regularization**

### **`dropout`**

* Applied inside the Transformer encoder and in the positional encoding block.
* **Typical values:** 0.1 – 0.3
* **Effect:** Regularizes to prevent overfitting.

### **Time Masking (augmentation)**

If enabled, these are additional knobs:

* **`timeMaskNum`**: number of time masks applied per sequence.
* **`timeMaskMaxFrac`**: max masked proportion of the sequence.

---

## **3. Positional Encoding Parameters**

### **`max_len`** (inside PositionalEncoding)

* Maximum sequence length supported by sinusoidal PE.
* **Default:** 5000
* **Tune if:** you change stride/kernel so your sequence becomes very long.

### **PE `dropout`**

* Same dropout as encoder; controls regularization after adding positions.

---

## **4. Sequence Resolution Parameters (affect Transformer input)**

These come from the original GRU architecture and remain tunable.
They control how many time steps the Transformer sees.

### **`strideLen`**

* Temporal stride in the Unfold operation.
* **Lower stride = more attention steps (larger sequence S)**
* **Higher stride = faster but less temporal resolution**

### **`kernelLen`**

* Size of the temporal window used by Unfold.
* Larger kernel smooths across time but increases input dimension.

### **`gaussianSmoothWidth`**

* Width of Gaussian smoothing applied before the Transformer.
* Helps denoise neural features.

---

## **5. Day-specific Adaptation Layers**

Retained from GRU-based model:

### **`nDays`**

* Number of recording days.
* Controls how many per-day transformations are learned.

---

## **6. Transformer Input Projection**

### **`input_proj`**

* Internal linear layer that maps each unfolded window
  (`neural_dim × kernelLen`) → `hidden_dim`.
* Usually not tuned directly, but its **input size** depends on:

  * `neural_dim`
  * `kernelLen`

---

# Recommended Starting Configs

### **Small Transformer**

```
hidden_dim: 256
layer_dim: 3
nhead: 4
dim_feedforward: 1024
dropout: 0.1
```

### **Medium Transformer (baseline comparison with GRU)**

```
hidden_dim: 512
layer_dim: 4
nhead: 8
dim_feedforward: 2048
dropout: 0.2
```

### **Large Transformer**

```
hidden_dim: 768
layer_dim: 6
nhead: 12
dim_feedforward: 3072
dropout: 0.1
```
