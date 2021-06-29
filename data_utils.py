import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_data_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        # Get images
        images = np.array(f['images'])
        labels = np.array(f['ans'])
        
        # Convert labels to 10 categorical classes
        labels = tf.keras.utils.to_categorical(labels, 10)
    return images, labels

def test_train_split(images, labels):
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.2)
    train_idx.tofile('./train_idx.csv',sep='\n')
    test_idx.tofile('./test_idx.csv',sep='\n')
    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

def image_prep(x):
    image = tf.cast(x['image'], tf.float32)
    image = image / 255
    grey_image = tf.reduce_mean(input_tensor=image, axis=2, keepdims=True)
    assert grey_image.shape[0] == 256
    assert grey_image.shape[1] == 256
    assert grey_image.shape[2] == 1
    aug_image = tf.identity(grey_image)
    return aug_image, x['label']

def make_tf_Dataset(train_images, train_labels, test_images, test_labels):
    train = tf.data.Dataset.from_tensor_slices({"image":train_images, "label":train_labels})
    test = tf.data.Dataset.from_tensor_slices({"image":test_images, "label":test_labels})

    train = train.map(
        lambda x: image_prep(x)
    ).cache().shuffle(100).batch(128)
    test = test.map(
        lambda x: image_prep(x)
    ).cache().batch(128)

    return train,test

def get_data(filename):
    images, labels = get_data_from_h5(filename)
    train_images, train_labels, test_images, test_labels = test_train_split(images, labels)
    train, test = make_tf_Dataset(train_images, train_labels, test_images, test_labels)
    return train, test
