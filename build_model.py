import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, InputLayer
from define_models import augLayers, convXpress

def get_model(rnd_seed):
    """Builds the ML MOdel.

    Args:
        rnd_seed: Seed for the random functions.

    Returns:
        The ML Model.
    """
    # Make new sequential model
    model = Sequential()
    # Add Input Layer
    model.add(InputLayer(input_shape=(256,256,1)))
    # Add augmentation Layers
    augLayers(model, 224)
    # Add ConvXpress
    model.add(convXpress(rnd_seed,(224,224,1),10))

    return model