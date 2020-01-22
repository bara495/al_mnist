def load_data(nr_of_elements=60000):
    """
    returns a specified amount of data
    """
    from keras.datasets import mnist
    from sklearn.model_selection import train_test_split

    seed_number = None # TODO: verify that it works 

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if nr_of_elements != 60000:
        x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size = nr_of_elements, random_state = seed_number)

    return (x_train, y_train), (x_test, y_test)

def label_data(nr_of_labels):
    """ 
    Function that simulates labeling of data and returns 
    a specified amount of labeled data
    """
    labels = None

    return labels

    