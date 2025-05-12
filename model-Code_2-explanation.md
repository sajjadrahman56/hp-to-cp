
### ✅ Step-by-Step Documentation of the MBTI Model Code (Model Code 2 )

---

#### **Step 1: Install Required Libraries**

```python
!pip install -q transformers datasets scikit-learn
```

**Purpose**: Installs Hugging Face Transformers, Datasets, and scikit-learn. These libraries are essential for model loading, tokenization, data handling, and evaluation.

---

#### **Step 2: Import Libraries**

```python
import re, numpy as np, tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
```

**Purpose**: Loads the necessary Python libraries for text processing, model training, and data manipulation.

---

#### **Step 3: Load Dataset**

```python
df = pd.read_csv(file_path)
```

**Purpose**: Reads the MBTI dataset from a CSV file into a pandas DataFrame.

---

#### **Step 4: Preprocess Text**

```python
def preprocess_text(text):
    ...
df['cleaned_posts'] = df['posts'].apply(preprocess_text)
```

**Purpose**: Cleans the input text by removing URLs, mentions, punctuation, and extra spaces. This helps the model focus on meaningful words.

---

#### **Step 5: Encode Labels**

```python
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['type'])
```

**Purpose**: Transforms MBTI types (like 'INTJ', 'ENFP') into numerical labels for classification.

---

#### **Step 6: Train-Test Split**

```python
train_texts, val_texts, train_labels, val_labels = train_test_split(...)
```

**Purpose**: Splits the dataset into training and validation sets for model training and evaluation.

---

#### **Step 7: Tokenize Text**

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, ...)
```

**Purpose**: Converts text into token IDs and attention masks, which are inputs BERT requires.

---

#### **Step 8: Create TensorFlow Datasets**

```python
train_dataset = tf.data.Dataset.from_tensor_slices((...)).batch(16)
```

**Purpose**: Creates batched datasets from encoded inputs and labels for efficient training in TensorFlow.

---

#### **Step 9: Load Pretrained BERT Model**

```python
model = TFBertForSequenceClassification.from_pretrained(..., num_labels=16)
```

**Purpose**: Loads a BERT model fine-tuned for sequence classification with 16 output classes (MBTI types).

---

#### **Step 10: Calculate Class Weights**

```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(...)
```

**Purpose**: Computes weights to give more importance to rare classes, helping the model not overfit on dominant types.

---

#### **Step 11: Define Training Components**

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
```

**Purpose**: Sets up the loss function and optimizer. `from_logits=True` is required because the model outputs raw logits.

---

#### **Step 12: Custom Training Loop with Class Weights**

```python
for epoch in range(EPOCHS):
    ...
    weighted_loss = tf.reduce_mean(loss * class_weights_batch)
```

**Purpose**: Trains the model using custom logic to include class weights. Applies loss and backpropagation per batch.

---

#### **Step 13: Use `tf.gather()` for Efficient Class Weighting**

```python
class_weights_batch = tf.gather(tf.constant(...), labels)
```

**Purpose**: Efficiently fetches the weight for each label in the batch, optimized for GPU/TPU.

---

#### **Step 14: Prediction Function**

```python
def predict(texts):
    ...
```

**Purpose**: Allows testing the model on new samples. Returns the predicted MBTI type.

---

#### **Step 15: Sample Predictions**

```python
sample_texts = ["I enjoy helping others..."]
print("Predicted MBTI types:", predict(sample_texts))
```

**Purpose**: Demonstrates how to use the model for inference with example personality descriptions.

---
