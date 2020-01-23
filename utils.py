from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner

'''
simple CNN on the MNIST dataset and gets 99.37% after ~45 epochs
'''
def mnist_cnn(nr_of_labeled_examples=60000, verbose=0):
    assert (nr_of_labeled_examples >= 100 and nr_of_labeled_examples <= 60000 and nr_of_labeled_examples % 10 == 0), \
        "Number of labeled example should be between 100 and 60000 and be dividible by 10"
    batch_size = 128
    num_classes = 10
    epochs = 100

    img_rows, img_cols = 28, 28

    (X_train, y_train), (x_test, y_test) = load_data(nr_of_labeled_examples)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255
    input_shape = (img_rows, img_cols, 1)

    if verbose == 2: 
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = create_model(input_shape)

    # added early stopping to avoid training when it's not progressing
    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.0001, patience=20, verbose=1, restore_best_weights=True)

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test), 
            callbacks=[es])

    score = model.evaluate(x_test, y_test, verbose=0)
    
    if verbose >= 1:
        print('Test loss:', score[0])
        print('Test accuracy:', score[1]) # TODO: change print, ? validation accuracy ?


    #AL!!
    n_initial = int(nr_of_labeled_examples / 10)
    # create the classifier
    classifier = KerasClassifier(create_model)
    
    # assemble initial data
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_initial = X_train[initial_idx]
    y_initial = y_train[initial_idx]

    print('\n\n', X_initial.shape, y_initial.shape, '\n\n')

    # generate the pool
    # remove the initial data from the training dataset
    X_pool = np.delete(X_train, initial_idx, axis=0)[:5000]
    y_pool = np.delete(y_train, initial_idx, axis=0)[:5000]

    print('\n\n', X_pool.shape, y_pool.shape, '\n\n')


    # initialize ActiveLearner
    learner = ActiveLearner(
        estimator=classifier,
        X_training=X_initial, y_training=y_initial,
        verbose=1
    )

    # the active learning loop
    n_queries = 9
    only_new = True # TODO: maybe learn on all data from the beggining, test!!!

    for idx in range(n_queries):
        print('Query no. %d' % (idx + 1))
        query_idx, query_instance = learner.query(X_pool, n_instances=n_initial, verbose=0)
        learner.teach(
            X=X_pool[query_idx], y=y_pool[query_idx], only_new=only_new,
            verbose=1
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

    return score

def create_model(input_shape=(28, 28, 1)):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

    return model


def load_data(nr_of_elements=60000):
    """
    returns a specified amount of data
    """
    seed_number = None # TODO: verify that it works 

    (X_train, y_train), (x_test, y_test) = mnist.load_data()

    if nr_of_elements != 60000:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size = nr_of_elements, random_state = seed_number)

    return (X_train, y_train), (x_test, y_test)

def label_data(nr_of_labels):
    """ 
    Function that simulates labeling of data and returns 
    a specified amount of labeled data
    """
    labels = None

    return labels

    