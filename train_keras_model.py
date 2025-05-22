import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Load the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Load the saved sign language model
model = tf.keras.models.load_model('keras_model.h5')  # Replace with your model path


# Preprocess each frame for prediction
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))  # Resize to match model input shape
    normalized_frame = resized_frame / 255.0  # Normalize pixel values between 0 and 1
    return normalized_frame


# List of sign names (ensure it matches the number of classes in the model)
sign_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Extend or modify as needed

cap = cv2.VideoCapture(0)  # Use appropriate camera index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Detect hand in the frame
    hands, _ = detector.findHands(frame)

    if hands:
        hand = hands[0]  # Assuming one hand is detected
        bbox = hand['bbox']

        # Extract hand region
        x, y, w, h = bbox
        hand_region = frame[y:y + h, x:x + w]

        # Ensure the hand region is valid for processing
        if hand_region.size != 0:
            processed_hand = preprocess_frame(hand_region)

            # Make prediction using the model
            prediction = model.predict(np.expand_dims(processed_hand, axis=0))
            predicted_class = np.argmax(prediction)

            # Validate index before accessing sign_names
            if 0 <= predicted_class < len(sign_names):
                sign_name = sign_names[predicted_class]
            else:
                sign_name = "Unknown"

            # Display the predicted sign on the frame
            cv2.putText(frame, f"Predicted sign: {sign_name}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Hand not fully visible", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Sign Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
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

# Save the loaded model in TensorFlow's SavedModel format (optional)
model_B = tf.keras.models.load_model('keras_model.h5')  # Replace with your model path
model_B.save('model_B_saved')  # Saves in TensorFlow's SavedModel format
print("Model B saved successfully.")
