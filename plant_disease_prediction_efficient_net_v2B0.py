import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Ensure TensorFlow uses the GPU if available
print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# List all physical devices available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data Preprocessing
AUTOTUNE = tf.data.AUTOTUNE

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

training_set = train_datagen.flow_from_directory(
    'M:\\machineLearning\\Plant_Disease_Prediction\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\train',
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator()

validation_set = validation_datagen.flow_from_directory(
    'M:\\machineLearning\\Plant_Disease_Prediction\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\valid',
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical'
)

# Convert DirectoryIterator to TensorFlow Dataset
def prepare(generator, shuffle=False):
    ds = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 38), dtype=tf.float32)
        )
    )
    ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(32)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = prepare(training_set, shuffle=True)
val_ds = prepare(validation_set)

# Building Model
base_model = EfficientNetV2B0(include_top=False, input_shape=(96, 96, 3), weights='imagenet')
base_model.trainable = True  # Unfreeze the base model for fine-tuning

# Fine-tuning: Unfreeze the top layers
for layer in base_model.layers[:-30]:  # Adjust the number of layers to unfreeze
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(38, activation='softmax', dtype='float32')
])

# Compiling and Training Phase
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Adding callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# Calculate steps per epoch
steps_per_epoch = (training_set.samples // training_set.batch_size) * 2  # Increase the number of steps per epoch
validation_steps = validation_set.samples // validation_set.batch_size

# Debugging information
print(f"Training steps per epoch: {steps_per_epoch}")
print(f"Validation steps per epoch: {validation_steps}")

# Training the first epoch to check for serialization issues
history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=1
)

# Saving Model after the first epoch to catch serialization errors early
try:
    model.save('trained_plant_disease_model_efficientnetv2b0_epoch1.keras')
except Exception as e:
    print(f"Error saving model after first epoch: {e}")
    raise e

# Continue training if no errors occur
training_history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=9,  # 9 more epochs to make a total of 10
    callbacks=callbacks
)

# Evaluating Model
train_loss, train_acc = model.evaluate(train_ds, steps=training_set.samples // training_set.batch_size)
print('Training accuracy:', train_acc)

val_loss, val_acc = model.evaluate(val_ds, steps=validation_set.samples // validation_set.batch_size)
print('Validation accuracy:', val_acc)

# Saving Model
model.save('trained_plant_disease_model_efficientnetv2b0.keras')

# Convert EagerTensor to float for JSON serialization
history = {key: [float(val) for val in values] for key, values in training_history.history.items()}

# Recording History in json
with open('training_hist.json', 'w') as f:
    json.dump(history, f)

print(training_history.history.keys())

# Accuracy and Loss Visualization
epochs = range(1, len(training_history.history['accuracy']) + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, training_history.history['loss'], color='red', label='Training Loss')
plt.plot(epochs, training_history.history['val_loss'], color='blue', label='Validation Loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Some other metrics for model evaluation
class_name = list(validation_set.class_indices.keys())
test_set = validation_datagen.flow_from_directory(
    'M:\\machineLearning\\Plant_Disease_Prediction\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\valid',
    target_size=(96, 96),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Reset test set generator
test_set.reset()

# Predict in batches
y_pred = model.predict(test_set, steps=validation_set.samples, verbose=1)
predicted_categories = np.argmax(y_pred, axis=1)
true_categories = test_set.classes

# Compute confusion matrix
cm = confusion_matrix(true_categories, predicted_categories)

# Print precision, recall, fscore
print(classification_report(true_categories, predicted_categories, target_names=class_name))

# Visualize Confusion Matrix
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 10})
plt.xlabel('Predicted Class', fontsize=20)
plt.ylabel('Actual Class', fontsize=20)
plt.title('Plant Disease Prediction Confusion Matrix', fontsize=25)
plt.show()
