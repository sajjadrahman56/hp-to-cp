# Continue from your preprocessing code...

# 6. CREATE TENSORFLOW DATASETS (MISSING IN YOUR CODE)
def create_dataset(encodings, labels):
    return tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'token_type_ids': encodings['token_type_ids']
        },
        labels
    ))

# Convert encoded data to TensorFlow datasets
train_dataset = create_dataset(train_encodings, train_df['encoded_labels'].values)
val_dataset = create_dataset(val_encodings, val_df['encoded_labels'].values)
test_dataset = create_dataset(test_encodings, test_df['encoded_labels'].values)

# Batch the datasets (add this)
batch_size = 32
train_dataset = train_dataset.shuffle(len(train_df)).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# 7. MODEL INITIALIZATION (YOUR CODE WITH IMPROVEMENTS)
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
from tensorflow.keras.callbacks import EarlyStopping

# Initialize model outside strategy
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=16)
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    config=config,
    from_pt=True
)

# Handle distribution strategy
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {strategy.num_replicas_in_sync} GPUs")
    with strategy.scope():
        # Rebuild classifier WITHIN strategy scope
        model.classifier = tf.keras.layers.Dense(
            16,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            activation=None,  # Explicitly set (important!)
            name='classifier'
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
except:
    print("Using single device")
    model.classifier = tf.keras.layers.Dense(
        16,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        activation=None,
        name='classifier'
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

# 8. TRAINING WITH IMPROVED CALLBACKS
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Increased from 2
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,  # Increased from 4
    callbacks=[early_stopping],
    verbose=1
)

# 9. EVALUATION (MISSING IN YOUR CODE)
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 10. SAVE MODEL (RECOMMENDED)
model.save_pretrained('./mbti_bert')
tokenizer.save_pretrained('./mbti_bert')
