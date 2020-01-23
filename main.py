from utils import mnist_cnn

# 60000 is the whole dataset, don't exceed that number
for i in [100, 1000, 10000, 60000]:
    print('\nComparing random sampling and the *aquisition function* for ', i, ' samples: \n')
    score = mnist_cnn(i)
    print('\tRandom sampling val_acc = ', score[1])
    # print('\t*Aquisition function val_acc = ', score[1], ' surpassed random sampling with ', quo_samples, ' samples.')
