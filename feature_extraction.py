import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import sys
# TODO: import Keras layers you need here

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', 'Bottleneck features training file(.p)')
flags.DEFINE_string('validation_file', '', 'Bottleneck features validation file(.p)')
flags.DEFINE_integer('epochs', 50, 'The number of epochs')
flags.DEFINE_integer('batch_size', 256, 'The batch size')

def load_bottleneck_data(training_file, validation_file):
    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val

def main(_):
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file,
                                                         FLAGS.validation_file)
    nb_classes = len(np.unique(y_train))
    #model
    inp_shape = X_train.shape[1:]
    inp = Input(shape = inp_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation = 'softmax')(x)
    model = Model(inp, x)
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    #train
    model.fit(X_train, y_train, epochs = FLAGS.epochs, batch_size = FLAGS.batch_size, validation_data = (X_val, y_val), shuffle= True)

if __name__ == '__main__':
    tf.app.run()
