import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from vit_keras import vit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


def create_vit_model(image_size, num_classes):
    model = vit.vit_b16(
        image_size=image_size,  # Expecting a single integer for ViT model
        activation='softmax',
        pretrained=False,
        include_top=True,
        pretrained_top=False,
        classes=num_classes
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_data_generators(data_directory, image_size):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    # Target size needs to be a tuple for the image data generator
    target_size = (image_size, image_size)  # Create a tuple (256, 256)

    train_generator = datagen.flow_from_directory(
        data_directory,
        target_size=target_size,  # Now passing a tuple
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_directory,
        target_size=target_size,  # Now passing a tuple
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator


def train_evaluate_save_model(model, train_generator, validation_generator):
    vit_history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )

    # Check if model file exists and remove it
    model_path = 'vit_model.h5'
    if os.path.exists(model_path):
        os.remove(model_path)  # Remove the existing model file

    model.save(model_path)  # Save the model

    vit_evaluation = model.evaluate(validation_generator)
    print("\nViT Model Evaluation:")
    print("Validation Loss:", vit_evaluation[0])
    print("Validation Accuracy:", vit_evaluation[1])

    plot_history(vit_history, 'ViT Model Training History')


def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Set your dataset path
    dataset_path = 'data'

    # Set your image size (use a single integer)
    image_size = 256  # Single integer for ViT model

    # Get the number of classes
    num_classes = len(os.listdir(dataset_path))

    # Create ViT model
    vit_model = create_vit_model(image_size, num_classes)

    # Create data generators
    train_generator, validation_generator = create_data_generators(dataset_path, image_size)

    # Train, evaluate, and save the model
    train_evaluate_save_model(vit_model, train_generator, validation_generator)
