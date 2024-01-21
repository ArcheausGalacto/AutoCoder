#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Ensure that TensorFlow version 2.x is used
assert tf.__version__.startswith('2.')

# Define constants
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)
NOISE_DIM = 100


# Utility function to download the TF flowers dataset if not already present
def download_data():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(
        origin=dataset_url, 
        fname='flower_photos', 
        untar=True
    )
    return data_dir


# Function to load and preprocess the flowers dataset
def load_flower_dataset(data_directory):
    dataset = image_dataset_from_directory(
        data_directory,
        label_mode=None,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    # Normalize the images to [-1, 1]
    dataset = dataset.map(lambda x: (x / 127.5) - 1)
    return dataset


# Function to create the generator model
def make_generator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8*8*256, use_bias=False, input_shape=(NOISE_DIM,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Reshape((8, 8, 256)),
        tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


# Function to create the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model


# Loss and optimizer functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Define the train step
@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Train function
def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GAN on the flowers dataset and generate synthetic data.')
    parser.add_argument('--data_dir', type=str, help='Path to the flowers dataset directory.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train GAN.')
    args = parser.parse_args()

    if args.data_dir is None or not os.path.exists(args.data_dir):
        print("Dataset directory not provided or does not exist. Downloading the dataset...")
        args.data_dir = download_data()

    print("Loading dataset...")
    dataset = load_flower_dataset(args.data_dir)

    print("Creating models...")
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    print(f"Starting training for {args.epochs} epochs...")
    train(dataset, args.epochs, generator, discriminator, generator_optimizer, discriminator_optimizer)

    print("Generating synthetic data...")
    noise = tf.random.normal([1, NOISE_DIM])
    generated_images = generator(noise, training=False)

    # Print generated data (the generated images as numpy arrays)
    np.set_printoptions(threshold=sys.maxsize)
    print(generated_images[0].numpy())