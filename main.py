from mnist_cnn import mnist_cnn

# 60000 is the whole dataset, don't exceed that number
for i in [100, 1000, 10000, 60000]:
    score = mnist_cnn(i)
    print(i, ' examples used, val_acc = ', score[1])