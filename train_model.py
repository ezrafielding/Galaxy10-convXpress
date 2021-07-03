import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import datetime
from data_utils import get_data
from build_model import get_model

def get_learning_rate_metric(optimizer):
    """Gets the learning rate metric.

    Args:
        optomizer: The optimizer for the model.

    Returns:
        The current learning rate metric.
    """
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

if __name__ == "__main__":
    # Get Dataset
    train, test = get_data('Galaxy10_DECals.h5')

    # Set Training parameters
    learning_rate = 1e-4
    epochs = 1000
    rnd_seed = 8901

    # Set Up optimizer and loss function
    optimizer = Adam(learning_rate=learning_rate)
    learningRateMetric = get_learning_rate_metric(optimizer)
    loss_function = CategoricalCrossentropy(from_logits=False)

    # Set Up Calls-backs
    model_checkpoint = ModelCheckpoint('./checkpoints/Galaxy10_convXpress_'+str(learning_rate)+'_'+str(epochs)+'.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-10, min_delta=0.001,mode='min')
    log_dir="./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop = EarlyStopping(restore_best_weights=True, patience=10)

    # Get and Compile Model
    model = get_model(rnd_seed)
    model.compile(optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy',learningRateMetric])
    print(model.summary())

    # Fit Model and Save
    hist = model.fit(train, validation_data=test, epochs=epochs,callbacks=[tensorboard_callback,model_checkpoint,early_stop,reduce_learning_rate],verbose=1)
    model.save('./model_save/'+'Galaxy10_convXpress_'+str(learning_rate)+'_'+str(epochs)+'_final.h5')

