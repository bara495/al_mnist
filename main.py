from utils import mnist_cnn

# 60000 is the whole dataset, don't exceed that number
for i in [100, 1000, 10000, 60000]:
    print('\nComparing random sampling and the *aquisition function* for ', i, ' samples: \n')
    mnist_cnn(i)
