import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import shutil

TRAIN_DIR_PARAM = '/path/to/dataset/train'
VALIDATION_DIR_PARAM = '/path/to/dataset/validation'
TEST_IMAGE_PARAM = '/path/to/test_image.jpg'
OUTPUT_DIR = 'output_results'

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE_EARLY_STOPPING = 10
PATIENCE_REDUCE_LR = 3
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_EPOCHS = 20
FINE_TUNE_AT_LAYER_PERCENT = 0.7 # Percentage of base model layers to keep frozen

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_dummy_data_if_needed(train_dir, validation_dir, test_img_path):
    dummy_created = False
    if not (os.path.exists(train_dir) and os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0):
        print(f"Warning: Training directory {train_dir} is invalid or empty. Creating dummy data.")
        shutil.rmtree(train_dir, ignore_errors=True)
        os.makedirs(os.path.join(train_dir, 'class_0_cat'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'class_1_dog'), exist_ok=True)
        dummy_created = True
    if not (os.path.exists(validation_dir) and os.path.isdir(validation_dir) and len(os.listdir(validation_dir)) > 0):
        print(f"Warning: Validation directory {validation_dir} is invalid or empty. Creating dummy data.")
        shutil.rmtree(validation_dir, ignore_errors=True)
        os.makedirs(os.path.join(validation_dir, 'class_0_cat'), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, 'class_1_dog'), exist_ok=True)
        dummy_created = True

    if dummy_created:
        from PIL import Image
        def create_dummy_image(path, color):
            img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=color)
            img.save(path)

        for i in range(BATCH_SIZE * 3):
            create_dummy_image(os.path.join(train_dir, 'class_0_cat', f'dummy_cat_train_{i}.jpg'), 'blue')
            create_dummy_image(os.path.join(train_dir, 'class_1_dog', f'dummy_dog_train_{i}.jpg'), 'green')
        for i in range(BATCH_SIZE * 2):
            create_dummy_image(os.path.join(validation_dir, 'class_0_cat', f'dummy_cat_val_{i}.jpg'), 'blue')
            create_dummy_image(os.path.join(validation_dir, 'class_1_dog', f'dummy_dog_val_{i}.jpg'), 'green')

    if not os.path.exists(test_img_path):
        print(f"Warning: Test image {test_img_path} not found. Creating a dummy test image.")
        dummy_test_dir = os.path.dirname(test_img_path)
        if not dummy_test_dir: dummy_test_dir = "."
        os.makedirs(dummy_test_dir, exist_ok=True)
        from PIL import Image
        img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='red')
        img.save(test_img_path)
        print(f"Created a dummy test image at: {test_img_path}")

create_dummy_data_if_needed(TRAIN_DIR_PARAM, VALIDATION_DIR_PARAM, TEST_IMAGE_PARAM)

train_datagen = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

validation_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR_PARAM,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR_PARAM,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

class_labels_map = {v: k for k, v in train_generator.class_indices.items()}
print(f"Class Indices: {train_generator.class_indices}")
print(f"Label Mapping for Prediction: {class_labels_map}")


base_model = EfficientNetB0(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                            include_top=False,
                            weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(name="avg_pool"),
    layers.BatchNormalization(),
    layers.Dropout(0.3, name="top_dropout"),
    layers.Dense(1, activation='sigmoid', name="output")
])

model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()

log_dir = os.path.join(OUTPUT_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_path = os.path.join(OUTPUT_DIR, 'best_feature_extraction_model.keras')
fine_tuned_model_path = os.path.join(OUTPUT_DIR, 'best_fine_tuned_model.keras')

callbacks_feature_extraction = [
    EarlyStopping(monitor='val_auc', mode='max', patience=PATIENCE_EARLY_STOPPING, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE_REDUCE_LR, min_lr=1e-6, verbose=1),
    tensorboard_callback
]

steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

print("\n--- Starting Feature Extraction Training ---")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_feature_extraction
)

def plot_training_history(history_obj, stage_name):
    acc = history_obj.history['accuracy']
    val_acc = history_obj.history['val_accuracy']
    loss = history_obj.history['loss']
    val_loss = history_obj.history['val_loss']
    auc_metric = history_obj.history.get('auc', [])
    val_auc_metric = history_obj.history.get('val_auc', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{stage_name} Training and Validation Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{stage_name} Training and Validation Loss')
    
    if auc_metric and val_auc_metric:
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, auc_metric, label='Training AUC')
        plt.plot(epochs_range, val_auc_metric, label='Validation AUC')
        plt.legend(loc='lower right')
        plt.title(f'{stage_name} Training and Validation AUC')

    plt.savefig(os.path.join(OUTPUT_DIR, f'{stage_name.lower().replace(" ", "_")}_history.png'))
    plt.show()

plot_training_history(history, "Feature Extraction")

print("\n--- Evaluating Feature Extraction Model ---")
if os.path.exists(model_checkpoint_path):
    print(f"Loading best model from: {model_checkpoint_path}")
    model = models.load_model(model_checkpoint_path)
else:
    print("Using model from last epoch of feature extraction.")

val_loss, val_accuracy, val_auc = model.evaluate(validation_generator, steps=validation_steps, verbose=1)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation AUC: {val_auc:.4f}")

Y_pred_probs = model.predict(validation_generator, steps=validation_steps, verbose=1)
Y_pred_binary = (Y_pred_probs > 0.5).astype(int).flatten()
Y_true = validation_generator.classes[:len(Y_pred_binary)]


print('\nConfusion Matrix')
cm = confusion_matrix(Y_true, Y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels_map.values(),
            yticklabels=class_labels_map.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Feature Extraction')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_feature_extraction.png'))
plt.show()

print('\nClassification Report')
target_names_sorted = [class_labels_map[i] for i in sorted(class_labels_map.keys())]
print(classification_report(Y_true, Y_pred_binary, target_names=target_names_sorted))

print('\nROC Curve')
fpr, tpr, thresholds = roc_curve(Y_true, Y_pred_probs.flatten())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Feature Extraction')
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_feature_extraction.png'))
plt.show()

def visualize_misclassified_images(model_to_test, data_generator, class_map, num_images=10):
    print("\n--- Visualizing Misclassified Images ---")
    misclassified_count = 0
    data_generator.reset()
    plt.figure(figsize=(15, num_images * 3 // 5 + 3))
    
    # Ensure we iterate enough batches to cover num_images or all samples
    num_batches_to_check = (data_generator.samples + data_generator.batch_size -1) // data_generator.batch_size

    for i in range(num_batches_to_check):
        if misclassified_count >= num_images:
            break
        x_batch, y_batch_true = next(data_generator)
        y_batch_pred_probs = model_to_test.predict(x_batch, verbose=0)
        y_batch_pred_labels = (y_batch_pred_probs > 0.5).astype(int).flatten()

        for j in range(len(y_batch_true)):
            if misclassified_count >= num_images:
                break
            true_label = int(y_batch_true[j])
            pred_label = y_batch_pred_labels[j]

            if true_label != pred_label:
                misclassified_count += 1
                plt.subplot(num_images // 5 + (1 if num_images % 5 > 0 else 0), 5, misclassified_count)
                
                img_to_show = x_batch[j]
                # EfficientNet preprocess_input scales to [-1, 1]. Revert for display.
                img_to_show = (img_to_show + 1) / 2.0 
                img_to_show = np.clip(img_to_show, 0, 1) # Clip to ensure valid range

                plt.imshow(img_to_show)
                plt.title(f"True: {class_map[true_label]}\nPred: {class_map[pred_label]} ({y_batch_pred_probs[j][0]:.2f})", fontsize=8)
                plt.axis('off')
    if misclassified_count == 0:
        print("No misclassified images found to display.")
    else:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'misclassified_images.png'))
        plt.show()

visualize_misclassified_images(model, validation_generator, class_labels_map)

print("\n--- Fine-tuning the Model ---")
base_model.trainable = True
num_layers_total = len(base_model.layers)
fine_tune_from_layer = int(num_layers_total * (1 - FINE_TUNE_AT_LAYER_PERCENT)) # Unfreeze top X%

print(f"Total layers in base model: {num_layers_total}")
print(f"Unfreezing from layer {fine_tune_from_layer} onwards.")

for layer in base_model.layers[:fine_tune_from_layer]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()
print(f"Number of trainable variables after unfreezing: {len(model.trainable_variables)}")

callbacks_fine_tuning = [
    EarlyStopping(monitor='val_auc', mode='max', patience=PATIENCE_EARLY_STOPPING + 5, restore_best_weights=True, verbose=1), # More patience
    ModelCheckpoint(filepath=fine_tuned_model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE_REDUCE_LR, min_lr=1e-7, verbose=1),
    tensorboard_callback
]

initial_epochs_feature_extraction = len(history.history['loss'])

history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=initial_epochs_feature_extraction + FINE_TUNE_EPOCHS,
    initial_epoch=initial_epochs_feature_extraction,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_fine_tuning
)

plot_training_history(history_fine_tune, "Fine-Tuning")

print("\n--- Evaluating Fine-Tuned Model ---")
if os.path.exists(fine_tuned_model_path):
    print(f"Loading best fine-tuned model from: {fine_tuned_model_path}")
    model = models.load_model(fine_tuned_model_path)
else:
    print("Using model from last epoch of fine-tuning.")

val_loss_ft, val_accuracy_ft, val_auc_ft = model.evaluate(validation_generator, steps=validation_steps, verbose=1)
print(f"Fine-tuned Validation Accuracy: {val_accuracy_ft*100:.2f}%")
print(f"Fine-tuned Validation Loss: {val_loss_ft:.4f}")
print(f"Fine-tuned Validation AUC: {val_auc_ft:.4f}")

Y_pred_probs_ft = model.predict(validation_generator, steps=validation_steps, verbose=1)
Y_pred_binary_ft = (Y_pred_probs_ft > 0.5).astype(int).flatten()
Y_true_ft = validation_generator.classes[:len(Y_pred_binary_ft)]

print('\nConfusion Matrix (Fine-Tuned)')
cm_ft = confusion_matrix(Y_true_ft, Y_pred_binary_ft)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ft, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels_map.values(),
            yticklabels=class_labels_map.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Fine-Tuned')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_fine_tuned.png'))
plt.show()

print('\nClassification Report (Fine-Tuned)')
print(classification_report(Y_true_ft, Y_pred_binary_ft, target_names=target_names_sorted))

print('\nROC Curve (Fine-Tuned)')
fpr_ft, tpr_ft, _ = roc_curve(Y_true_ft, Y_pred_probs_ft.flatten())
roc_auc_ft = auc(fpr_ft, tpr_ft)

plt.figure()
plt.plot(fpr_ft, tpr_ft, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_ft:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Fine-Tuned')
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_fine_tuned.png'))
plt.show()

def predict_single_image(model_to_use, img_path, class_map):
    if not os.path.exists(img_path):
        print(f"Error: Test image not found at {img_path}")
        return

    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = efficientnet_preprocess_input(img_array_expanded)
    
    prediction_prob = model_to_use.predict(img_array_preprocessed, verbose=0)[0][0]
    
    predicted_class_index = int(prediction_prob > 0.5)
    predicted_class_label = class_map.get(predicted_class_index, "Unknown")
    confidence = prediction_prob if predicted_class_index == 1 else 1 - prediction_prob
    
    print(f"\n--- Prediction for {os.path.basename(img_path)} ---")
    print(f"Predicted as: {predicted_class_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Raw probability (for class 1): {prediction_prob:.4f}")

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class_label} ({confidence:.2f})")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_{os.path.basename(img_path)}.png'))
    plt.show()

print("\n--- Final Prediction with Best Fine-Tuned Model ---")
final_model_to_use = model # This is already the best fine-tuned model due to restore_best_weights or loaded from checkpoint
if os.path.exists(fine_tuned_model_path):
    print(f"Using model from {fine_tuned_model_path} for final prediction.")
    final_model_to_use = models.load_model(fine_tuned_model_path)
elif os.path.exists(model_checkpoint_path):
     print(f"Fine-tuned model not found, using best feature extraction model from {model_checkpoint_path} for final prediction.")
     final_model_to_use = models.load_model(model_checkpoint_path)
else:
    print("No saved model found. Using model from last training epoch.")


predict_single_image(final_model_to_use, TEST_IMAGE_PARAM, class_labels_map)

print(f"\nAll results, logs, and plots saved in: {os.path.abspath(OUTPUT_DIR)}")
print("Script finished.")