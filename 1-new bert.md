Creating a new virtual environment (venv) can often resolve dependency conflicts and version-related issues, but you'll need to follow specific steps to ensure it works for your BERT classification task. Here's a complete guide:

### Step-by-Step Solution with New Virtual Environment

1. **Create and activate a new venv**:
```bash
python -m venv bert_env
source bert_env/bin/activate  # Linux/Mac
# or
bert_env\Scripts\activate  # Windows
```

2. **Install compatible versions**:
```bash
pip install tensorflow==2.12.0
pip install transformers==4.30.0
pip install pandas scikit-learn matplotlib
```

3. **Modified Code for Venv**:
```python
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
from tensorflow.keras.callbacks import EarlyStopping

# Initialize model OUTSIDE any distribution strategy first
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=16)
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    config=config,
    from_pt=True  # Critical for weight conversion
)

# Then handle strategy if needed
try:
    strategy = tf.distribute.MirroredStrategy()
    print("Using MirroredStrategy with", strategy.num_replicas_in_sync, "GPUs")
    with strategy.scope():
        # Rebuild classifier layer within strategy scope
        model.classifier = tf.keras.layers.Dense(
            16,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='classifier'
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(2e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
except:
    print("Using single GPU/CPU")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

# Training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=4,
    callbacks=[EarlyStopping(patience=2)]
)
```

### Key Advantages of New Venv:

1. **Clean Slate**:
   - Eliminates all previous conflicting installations
   - Ensures no hidden package interactions

2. **Version Control**:
   - The specified versions (TF 2.12 + Transformers 4.30) are known to work well together
   - Avoids issues with newer Keras 3.x changes

3. **Isolation**:
   - Prevents system-wide Python packages from interfering
   - Makes dependency management reproducible

### Additional Verification Steps:

1. Check installations:
```python
import tensorflow as tf
from transformers import __version__ as transformers_version
print("TF version:", tf.__version__)
print("Transformers version:", transformers_version)
```

2. Validate GPU access:
```python
print("GPU available:", tf.config.list_physical_devices('GPU'))
```

3. Test basic BERT operations:
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_input = tokenizer("Test input", return_tensors='tf')
print(model(test_input).logits.shape)  # Should output (1, 16)
```

### If Problems Persist:

1. Try downgrading further:
```bash
pip install tensorflow==2.10.0 transformers==4.25.1
```

2. Or use the CPU-only fallback:
```python
with tf.device('/CPU:0'):
    model.fit(...)  # Force CPU execution
```

This approach gives you the highest probability of success by:
- Starting with a clean environment
- Using battle-tested version combinations
- Providing proper fallbacks for different hardware configurations
- Maintaining all BERT functionality while avoiding distribution strategy issues
