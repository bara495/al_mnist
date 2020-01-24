from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling
import os
import time


'''
simple CNN on the MNIST dataset and gets 99.37% after ~45 epochs
'''
def mnist_cnn(nr_of_labeled_examples=60000, verbose=0):
    assert (nr_of_labeled_examples >= 100 and nr_of_labeled_examples <= 60000 and nr_of_labeled_examples % 10 == 0), \
        "Number of labeled example should be between 100 and 60000 and be dividible by 10"

    batch_size = 128
    epochs = 100

    model_path = 'best_model.h5'

    (X_train, y_train), (x_test, y_test) = load_proc_data(nr_of_labeled_examples)

    if verbose == 2: 
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    model = create_model()

    # added early stopping to avoid training when it's not progressing
    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.0001, patience=20, verbose=1, restore_best_weights=True)
    mc = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', save_best_only=True)

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test), 
            callbacks=[mc, es])

    i=0
    while True:
        if not os.path.exists(model_path):
            if i==0:
                print('Waiting h5 model file...')
            if i == 10:
                msg = 'error, check in logs'
                with open(f'result_{nr_of_labeled_examples}.txt', 'a') as f:
                    print(f'Random sampling - Training on {nr_of_labeled_examples} samples\n\tVal. accuracy: {msg}', file=f)
                break
            i = i + 1
            time.sleep(0.0001)
        else: 
            saved_model = load_model(model_path)
            score = saved_model.evaluate(x_test, y_test, verbose=0)
            os.remove(model_path)
            with open(f'result_{nr_of_labeled_examples}.txt', 'a') as f:
                print(f'Random sampling - Training on {nr_of_labeled_examples} samples\n\tVal. accuracy: ', '%.5f' % score[1], file=f)
            break
    
    if verbose >= 1:
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    #AL!!
    sampling_methods = [uncertainty_sampling, entropy_sampling, margin_sampling]

    
    for method in sampling_methods:
        with open(f'result_{nr_of_labeled_examples}.txt', 'a') as f:
                print(f'{method.__name__} - # trained samples    - Val. accuracy', file=f)

        (X_train, y_train), (x_test, y_test) = load_proc_data()

        segment = int(nr_of_labeled_examples / 10)
        # create the classifier
        classifier = KerasClassifier(create_model)

        # assemble initial data
        initial_idx = np.random.choice(range(len(X_train)), size=segment, replace=False)
        X_initial = X_train[initial_idx]
        y_initial = y_train[initial_idx]

        # initialize ActiveLearner
        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=method,
            X_training=X_initial, y_training=y_initial,
            verbose=1
        )

        # the active learning loop
        n_queries = 9
        only_new = False # TODO: maybe learn on all data from the beggining, test!!!

        for idx in range(n_queries):
            model_path_al = f'best_model_al_{method.__name__}_{(idx + 2)*segment}.h5'

            mc_al = ModelCheckpoint(model_path_al, monitor='val_accuracy', mode='max', save_best_only=True)
            
            print('Query no. %d' % (idx + 1))
            query_idx, _ = learner.query(X_train, n_instances=segment, verbose=0) #TODO: n_instances param, get it here somehow, or do the process for n times
            learner.teach(
                X=X_train[query_idx], 
                y=y_train[query_idx], 
                only_new=only_new,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test), 
                callbacks=[mc_al, es]
            )

            i=0
            while True:
                if not os.path.exists(model_path_al):
                    if i==0:
                        print('Waiting h5 model file...')
                    if i==10:
                        msg = 'error, check in logs'
                        with open(f'result_{nr_of_labeled_examples}.txt', 'a') as f:
                            print(f'                            {(idx + 2)*segment}            \t\t{msg}', file=f)
                        break
                    i = i + 1
                    time.sleep(0.01)
                else: 
                    saved_model = load_model(model_path_al)
                    score_al = saved_model.evaluate(x_test, y_test, verbose=0)
                    os.remove(model_path_al)
                    with open(f'result_{nr_of_labeled_examples}.txt', 'a') as f:
                        print(f'                            {(idx + 2)*segment}            \t\t','%.5f' % score_al[1], file=f)
                    break

            # remove queried instance from pool
            X_train = np.delete(X_train, query_idx, axis=0)
            y_train = np.delete(y_train, query_idx, axis=0)

            # score_al = learner.score_al(x_test, y_test)

    with open(f'result_{nr_of_labeled_examples}.txt', 'a') as f:
        print('\n', file=f)

    return

def create_model():
    input_shape=(28, 28, 1)

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


def load_proc_data(nr_of_elements=60000):
    """
    returns a specified amount of data
    """
    seed_number = None # TODO: verify that it aorks 
    img_rows, img_cols = 28, 28
    num_classes = 10


    (X_train, y_train), (x_test, y_test) = mnist.load_data()

    if nr_of_elements != 60000:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size = nr_of_elements, random_state = seed_number)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (x_test, y_test)

    