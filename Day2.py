import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# --- Configuration ---
BATCH_SIZE = 128
EPOCHS = 20 # Increased epochs, but EarlyStopping will prevent overfitting
PATIENCE_EARLY_STOPPING = 5
PATIENCE_REDUCE_LR = 3
MODEL_SAVE_PATH = 'best_mnist_cnn_model.keras'

# --- Data Loading ---
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# --- Preprocessing ---
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

y_train_categorical = to_categorical(train_labels)
y_test_categorical = to_categorical(test_labels)

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # featurewise_center=True, # Requires datagen.fit(train_images)
    # featurewise_std_normalization=True # Requires datagen.fit(train_images)
)
# If using featurewise operations, uncomment the line below
# datagen.fit(train_images)

# --- Model Building ---
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256)) # Increased dense layer size
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# --- Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE_EARLY_STOPPING,
    restore_best_weights=False, # We will load the best weights manually from checkpoint
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=PATIENCE_REDUCE_LR,
    min_lr=0.00001,
    verbose=1
)

callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

# --- Model Compilation ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Model Training ---
print("\n--- Starting Training ---")
history = model.fit(
    datagen.flow(train_images, y_train_categorical, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(test_images, y_test_categorical),
    steps_per_epoch=len(train_images) // BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)
print("--- Training Finished ---")

# --- Load Best Model and Evaluate ---
print("\n--- Loading Best Model from Checkpoint ---")
if os.path.exists(MODEL_SAVE_PATH):
    best_model = models.load_model(MODEL_SAVE_PATH)
    print("Best model loaded successfully.")

    print("\n--- Evaluating Best Model on Test Data ---")
    test_loss, test_acc = best_model.evaluate(test_images, y_test_categorical, verbose=0)
    print(f"Test Loss (Best Model): {test_loss:.4f}")
    print(f"Test Accuracy (Best Model): {test_acc * 100:.2f}%")

    # --- Predictions and Detailed Metrics ---
    print("\n--- Generating Predictions and Detailed Metrics ---")
    predictions = best_model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_labels # Use original integer labels

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=[str(i) for i in range(10)]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    print(f"\nPrediction for first test image: {predicted_classes[0]}")
    print(f"Actual label for first test image: {true_classes[0]}")

else:
    print(f"Error: Model checkpoint file not found at {MODEL_SAVE_PATH}. Evaluation skipped.")
    print("\n--- Evaluating Final Model State (Not Necessarily Best) ---")
    test_loss, test_acc = model.evaluate(test_images, y_test_categorical, verbose=0)
    print(f"Test Loss (Final State): {test_loss:.4f}")
    print(f"Test Accuracy (Final State): {test_acc * 100:.2f}%")


# --- Plotting Training History ---
print("\n--- Plotting Training History ---")
if history:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()