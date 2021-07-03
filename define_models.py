import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomFlip, RandomCrop
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, InputLayer
from tensorflow.keras import initializers, regularizers

def augLayers(model, crop_size):
    """Defines aurgmentation layers.

    Args:
        model: The model which the layers need to be added to.
        crop_size: The size of the crop for the RandomCrop Layer.

    Returns:
        Augmentation Layers.
    """
    model.add(RandomRotation(0.5, fill_mode='reflect'))
    model.add(RandomFlip())
    model.add(RandomCrop(crop_size, crop_size))

def convXpress(random_state,input_shape,num_classes):
    """Defines the ConvXpress Model.

    Args:
        random_state: Seed for the Random function.
        input_shape: The expected input shape for the model.
        num_classes: The size of the output layer / nmumber of classes for dataset.

    Returns:
        ConvXpress Layers.
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model