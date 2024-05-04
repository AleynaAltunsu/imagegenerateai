import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Define the generator model
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(7*7*128, input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='tanh'))
    return model

# Generate random points in the latent space
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Use the generator to generate n_samples of abstract art
def generate_images(generator, latent_dim, n_samples):
    latent_points = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(latent_points)
    return X

# Plot the generated images
def plot_generated_images(images, n_samples):
    for i in range(n_samples):
        plt.subplot(1, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(images[i, :, :, 0], cmap='gray_r')
    plt.show()

# Define the size of the latent space
latent_dim = 100

# Build and compile the generator
generator = build_generator(latent_dim)

# Generate and plot images
n_samples = 5
generated_images = generate_images(generator, latent_dim, n_samples)
plot_generated_images(generated_images, n_samples)
