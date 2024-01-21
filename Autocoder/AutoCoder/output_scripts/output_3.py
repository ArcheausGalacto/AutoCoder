#!/usr/bin/env python3
import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Utility function to download the TF flowers dataset if not already present
def download_data():
    import pathlib
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                       fname='flower_photos', 
                                       untar=True)
    data_dir = pathlib.Path(data_dir)
    return str(data_dir)

# Function to load the flower dataset
def load_flower_dataset(data_directory):
    """Load and preprocess the TF flowers dataset."""
    # Check if the given directory exists
    if not os.path.isdir(data_directory):
        # Download the dataset if the directory does not exist
        data_directory = download_data()
    
    dataset = image_dataset_from_directory(
        data_directory,
        label_mode=None,
        image_size=(64, 64),
        batch_size=32).map(lambda x: x / 255.0)
    return dataset

# Define the generator model 
def make_generator_model():
    """Creates the generator model that takes a seed (noise) and produces an image."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Reshape((8, 8, 256)),
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Define the discriminator model
def make_discriminator_model():
    """Creates the discriminator model that classifies real images from fake."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model

# Command line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GAN on the flowers dataset and generate images.')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Directory of the flowers dataset. If empty, dataset will be downloaded.')
    parser.add_argument('--generate', action='store_true',
                        help='Generate a flower image after training.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train the GAN.')

    args = parser.parse_args()

    # Load the dataset
    dataset = load_flower_dataset(args.data_dir if args.data_dir else 'path/to/flowers_dataset')

    # Create Generator and Discriminator
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define generator loss
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    # Define discriminator loss
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Note: This script does not include the training loop and the actual generation of fake data.
    # Below you would include additional code for training the GAN and generating the fake data
    # after the training process. This is just a placeholder and would need to be replaced
    # with actual training code that updates the generator and discriminator weights.

    if args.generate:
        # Generate and display a flower image after training
        noise = tf.random.normal([1, 100])
        generated_image = generator(noise, training=False)
        plt.imshow(generated_image[0, :, :, :], cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        print("Training mode is not implemented in this script.")