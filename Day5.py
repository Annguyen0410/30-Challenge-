import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(train_images)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_checkpoint_cb = ModelCheckpoint(
    filepath="best_mnist_model.keras",
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

early_stopping_cb = EarlyStopping(
    monitor='val_loss',
    patience=5, # Increased patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

callbacks_list = [model_checkpoint_cb, early_stopping_cb, reduce_lr_cb]

EPOCHS = 20 # Increased epochs
BATCH_SIZE = 64

history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(test_images, test_labels),
    steps_per_epoch=len(train_images) // BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

print("\nEvaluating the model with the best weights restored by EarlyStopping...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test accuracy: {test_acc * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_labels, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nClassification Report:")
print(classification_report(test_labels, predicted_classes, target_names=[str(i) for i in range(10)]))

def display_sample_predictions(images, true_labels, model_predictions, num_examples=10):
    plt.figure(figsize=(15, int(1.5 * num_examples))) # Adjusted for potentially more rows
    indices = np.random.choice(len(images), num_examples, replace=False)
    
    for i, index in enumerate(indices):
        plt.subplot((num_examples + 2) // 3, 3, i + 1) # Dynamic grid: 3 columns
        plt.imshow(images[index].reshape(28, 28), cmap='gray')
        true_label = true_labels[index]
        predicted_label = np.argmax(model_predictions[index])
        title_color = 'green' if predicted_label == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {predicted_label}", color=title_color)
        plt.axis('off')
    plt.suptitle("Sample Predictions (Green: Correct, Red: Incorrect)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

print("\nDisplaying some sample predictions from the test set:")
display_sample_predictions(test_images, test_labels, predictions, num_examples=12)


misclassified_indices = np.where(predicted_classes != test_labels)[0]
if len(misclassified_indices) > 0:
    print(f"\nFound {len(misclassified_indices)} misclassified images. Displaying some examples:")
    num_to_display = min(len(misclassified_indices), 9)
    
    plt.figure(figsize=(12, 12))
    display_indices = np.random.choice(misclassified_indices, num_to_display, replace=False)

    for i, index in enumerate(display_indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
        plt.title(f"True: {test_labels[index]}, Pred: {predicted_classes[index]}")
        plt.axis('off')
    plt.suptitle("Misclassified Examples", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
else:
    print("\nNo misclassified examples found in the test set!")

print("\nTo load the best saved model later (if needed):")
print("from tensorflow.keras.models import load_model")
print("loaded_model = load_model('best_mnist_model.keras')")