import tensorflow as tf
from tensorflow.keras import layers, models
import os

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    # Placeholder for dataset directory
    train_dir = '../data/train'
    val_dir = '../data/val'

    img_height, img_width = 64, 64
    batch_size = 32
    num_classes = 10  # Change according to your dataset

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    model = build_model((img_height, img_width, 3), num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Save model
    model.save('../models/cnn_image_classifier')
    print("Model saved at ../models/cnn_image_classifier")

if __name__ == '__main__':
    main()
