import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define constants
batch_size = 32
image_size = (64, 64)
dataset_path = 'data'
num_classes = 2

# Load CNN model
cnn_model = tf.keras.models.load_model('keras_model.h5')

# Load ELU model
elu_model = load_model('elu_cnn_model.h5')

# Load VGG model
vgg_model = load_model('vgg_model.h5')

# Create ImageDataGenerator for data normalization
test_datagen = ImageDataGenerator(rescale=1./255)

# Specify the test directory
test_directory = 'data'

# Create a test generator
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # Adjust based on your model's requirements
)

# Check the length of the generator
print("Number of test samples:", len(test_generator))

# Evaluate the models
elu_evaluation = elu_model.evaluate(test_generator)
vgg_evaluation = vgg_model.evaluate(test_generator)
cnn_evaluation = cnn_model.evaluate(test_generator)

# Print evaluation results
print("\nELU Model Evaluation:")
print("Validation Loss:", elu_evaluation[0])
print("Validation Accuracy:", elu_evaluation[1])

print("\nVGG Model Evaluation:")
print("Validation Loss:", vgg_evaluation[0])
print("Validation Accuracy:", vgg_evaluation[1])

print("\nCNN Model Evaluation:")
print("Validation Loss:", cnn_evaluation[0])
print("Validation Accuracy:", cnn_evaluation[1])

# Function to plot training history
def plot_training_history(history, title):
    if 'loss' in history:
        # Plot training and validation loss
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    if 'accuracy' in history:
        # Plot training and validation accuracy
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title(title + ' - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

# Example usage:
# Check if models have training history available
if hasattr(elu_model, 'history') and elu_model.history:
    plot_training_history(elu_model.history.history, 'ELU Model Training History')

if hasattr(vgg_model, 'history') and vgg_model.history:
    plot_training_history(vgg_model.history.history, 'VGG Model Training History')

if hasattr(cnn_model, 'history') and cnn_model.history:
    plot_training_history(cnn_model.history.history, 'CNN Model Training History')

# Compare validation accuracies
accuracies = [elu_evaluation[1], vgg_evaluation[1], cnn_evaluation[1]]
models = ['ELU Model', 'VGG Model', 'CNN Model']

plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Validation Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Function to count the number of parameters in a model
def count_parameters(model):
    return sum([tf.keras.backend.count_params(p) for p in set(model.trainable_weights)])

# Display the number of parameters for each model
elu_params = count_parameters(elu_model)
vgg_params = count_parameters(vgg_model)
cnn_params = count_parameters(cnn_model)

print("Number of Parameters:")
print("ELU Model:", elu_params)
print("VGG Model:", vgg_params)
print("CNN Model:", cnn_params)
