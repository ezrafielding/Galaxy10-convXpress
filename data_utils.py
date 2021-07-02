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
    # Get Train/Test Split
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.2)
    # Save Train and Test Indexes
    train_idx.tofile('./train_idx.csv',sep='\n')
    test_idx.tofile('./test_idx.csv',sep='\n')
    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

def image_prep(x):
    # Convert Images to tf.float32 data type
    image = tf.cast(x['image'], tf.float32)
    # Normalize pixel values
    image = image / 255
    # Remove clour by taking mean over channels
    grey_image = tf.reduce_mean(input_tensor=image, axis=2, keepdims=True)
    # Ensuring dimensions are correct
    assert grey_image.shape[0] == 256
    assert grey_image.shape[1] == 256
    assert grey_image.shape[2] == 1
    # Ensure new tensor is returned
    aug_image = tf.identity(grey_image)
    return aug_image, x['label']

def make_tf_Dataset(train_images, train_labels, test_images, test_labels):
    # Pack Images and Labels into dataset
    train = tf.data.Dataset.from_tensor_slices({"image":train_images, "label":train_labels})
    test = tf.data.Dataset.from_tensor_slices({"image":test_images, "label":test_labels})

    # Chache, shuffle and batch dataset. Image pre-processing also done
    train = train.map(
        lambda x: image_prep(x)
    ).cache().shuffle(100).batch(128)
    test = test.map(
        lambda x: image_prep(x)
    ).cache().batch(128)

    return train,test

def get_data(filename):
    # Fetch Images and labels
    images, labels = get_data_from_h5(filename)
    # Get Train/Test Split
    train_images, train_labels, test_images, test_labels = test_train_split(images, labels)
    # Pack data into datasets and pre-process images
    train, test = make_tf_Dataset(train_images, train_labels, test_images, test_labels)
    return train, test
