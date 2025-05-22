import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ELU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define constants
batch_size = 32
image_size = (64, 64)  # Modify based on your image dimensions
dataset_path = 'data'  # Replace with your dataset path
num_classes =10  # Adjust based on the number of classes in your dataset (e.g., '2' classes)

# Create ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load and prepare train and validation datasets
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create CNN model with ELU activation
model = Sequential([
    Conv2D(64, (3, 3), input_shape=(image_size[0], image_size[1], 3)),
    ELU(),  # Exponential Linear Unit
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3)),
    ELU(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3)),
    ELU(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512),
    ELU(),
    Dense(num_classes, activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]  # Using callbacks for early stopping and model saving
)

# Save the final model (even if it's not the best)
model.save('final_cnn_model.h5')
