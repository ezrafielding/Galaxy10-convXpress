import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from data_utils import get_data
from build_model import get_model

def get_learning_rate_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

if __name__ == "__main__":
    train, test = get_data('Galaxy10_DECals.h5')

    learning_rate = 1e-4
    epochs = 1000
    rnd_seed = 8901

    optimizer = Adam(learning_rate=learning_rate)
    learningRateMetric = get_learning_rate_metric(optimizer)
    loss_function = CategoricalCrossentropy(from_logits=False)

    model_checkpoint = ModelCheckpoint('Galaxy10_convXpress_'+str(learning_rate)+'_'+str(epochs)+'.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-10, min_delta=0.001,mode='min')
    tensorboard_callback = TensorBoard(log_dir="./logs")
    early_stop = EarlyStopping(restore_best_weights=True, patience=10)

    model = get_model(rnd_seed)
    model.compile(optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy',learningRateMetric])
    print(model.summary())

    hist = model.fit(train, validation_data=test, epochs=epochs,callbacks=[tensorboard_callback,model_checkpoint,early_stop,reduce_learning_rate],verbose=1)
    model.save('./model_save/'+'Galaxy10_convXpress_'+str(learning_rate)+'_'+str(epochs)+'_final.h5')

