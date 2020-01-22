from mnist_cnn import mnist_cnn

for i in [100, 1000, 10000, None]:
    score = mnist_cnn(i)
    print(str(60000 if i==None else i), ' examples used, val_acc = ', score[1])