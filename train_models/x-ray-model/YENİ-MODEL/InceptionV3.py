import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, AveragePooling2D, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import cv2

# تحميل نموذج InceptionV3
model = InceptionV3(weights='imagenet')

# Constants
IM_WIDTH, IM_HEIGHT = 299, 299
BATCH_SIZE = 32
EPOCHS = 10
FC_SIZE = 1024
WEIGHTS_FILE = "weights_3.h5"

# Function to count files in a directory
def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    return sum([len(files) for r, d, files in os.walk(directory)])

# Function to set up transfer learning
def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Function to add new layers on top of the base model
def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = Dropout(0.5)(x)
    x = AveragePooling2D((8, 8), name="avg_pool", padding="valid")(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Plot training results
def plot_training(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'r.', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')

    plt.figure()
    plt.plot(epochs, loss, 'b.', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss.png')

# Training function
def train(epochs, output_model_file, plot=True):
    train_dir = 'train/'
    val_dir = 'val/'

    nb_train_samples = get_nb_files(train_dir)
    nb_val_samples = get_nb_files(val_dir)
    nb_classes = len(glob.glob(train_dir + "/*"))

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    input_tensor = Input(shape=(IM_WIDTH, IM_HEIGHT, 3))
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
    model = add_new_last_layer(base_model, nb_classes)

    setup_to_transfer_learn(model, base_model)

    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=1, verbose=1)
    checkpoint = ModelCheckpoint(output_model_file, monitor='val_accuracy', save_best_only=True, verbose=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_val_samples // BATCH_SIZE,
        callbacks=[lr_reduce, checkpoint]
    )

    model.save(output_model_file)

    if plot:
        plot_training(history)

# Prediction functions
def predict(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

def plot_preds(preds):
    labels = ("NORMAL", "PNEUMONIA")
    plt.barh(range(len(labels)), preds, alpha=0.5)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Probability')
    plt.tight_layout()
    plt.savefig('predictions.png')

# Train the model
train(EPOCHS, WEIGHTS_FILE)

# Load the saved model
model = load_model(WEIGHTS_FILE)

# Example predictions
img_path = 'test/PNEUMONIA/person48_virus_100.jpeg'
preds = predict(model, img_path)
plot_preds(preds)
print("Prediction for first image:", np.argmax(preds))

img_path = 'test/NORMAL/IM-0105-0001.jpeg'
preds = predict(model, img_path)
plot_preds(preds)
print("Prediction for second image:", np.argmax(preds))
