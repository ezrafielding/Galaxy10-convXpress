import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, InputLayer
from define_models import augLayers, convXpress

def get_model(rnd_seed):
    model = Sequential()
    model.add(InputLayer(input_shape=(256,256,1)))
    augLayers(model, 224)
    model.add(convXpress(rnd_seed,(224,224,1),10))

    return model