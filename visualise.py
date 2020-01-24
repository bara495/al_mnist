import matplotlib.pyplot as plt

x100 = [20, 30, 40, 50, 60, 70, 80, 90, 100] 

a100 = 0.80420
b100 = [0.43790,0.49230, 0.55860, 0.57460, 0.60790, 0.63450, 0.67290, 0.70140, 0.70810]
c100 = [0.33850, 0.40210, 0.46010, 0.47910, 0.56880, 0.66980, 0.62870, 0.69290, 0.64670]
d100 = [0.49230, 0.58570, 0.63460, 0.71670, 0.71900, 0.78500, 0.81090, 0.82670, 0.85800]

all100 = [b100, c100, d100]
sampling_methods = ['uncertainty_sampling', 'entropy_sampling', 'margin_sampling']


plt.hlines(a100, x100[0], x100[-1], label='Random sampling - baseline')
for i in range(3):
    plt.plot(x100, all100[i], label=sampling_methods[i])
plt.title('100 samples')
plt.xlabel('Number of samples trained on')
plt.ylabel('Validation accuracy')
plt.legend()
plt.savefig('100.png')

plt.clf()

x1000 = [200, 300, 400, 500, 600, 700, 800, 900, 1000] 

a1000 = 0.95700

b1000 = [0.82420, 0.87720, 0.92090, 0.94570, 0.95440, 0.96430, 0.97280, 0.97670, 0.97830]
c1000 = [0.75690, 0.84610, 0.88210, 0.92200, 0.94650, 0.95470, 0.96160, 0.96870, 0.97140]
d1000 = [0.84380, 0.91230, 0.94980, 0.95750, 0.96520, 0.97380, 0.97490, 0.97580, 0.97980]

all1000 = [b1000, c1000, d1000]

plt.hlines(a1000, x1000[0], x1000[-1], label='Random sampling - baseline')
for i in range(3):
    plt.plot(x1000, all1000[i], label=sampling_methods[i])
plt.title('1000 samples')
plt.xlabel('Number of samples trained on')
plt.ylabel('Validation accuracy')
plt.legend()
plt.savefig('1000.png')

plt.clf()



x10000 = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] 

a10000 = 0.98670

b10000 = [0.97560, 0.98570, 0.99080, 0.99170, 0.99130, 0.99290, 0.99310, 0.99310, 0.99330]
c10000 = [0.97040, 0.98620, 0.99080, 0.99200, 0.99280, 0.99280, 0.99290, 0.99350, 0.99360]
d10000 = [0.97470, 0.98610, 0.99100, 0.99280, 0.99290, 0.99320, 0.99410, 0.99300, 0.99370]

all10000 = [b10000, c10000, d10000]

plt.hlines(a10000, x10000[0], x10000[-1], label='Random sampling - baseline')
for i in range(3):
    plt.plot(x10000, all10000[i], label=sampling_methods[i])
plt.title('10000 samples')
plt.xlabel('Number of samples trained on')
plt.ylabel('Validation accuracy')
plt.legend()
plt.savefig('10000.png')

plt.clf()


x60000 = [12000, 18000, 24000, 30000, 36000, 42000, 48000, 54000, 60000] 

a60000 = 0.99400

b60000 = [0.99180, 0.99350, 0.99410, 0.99410, 0.99360, 0.99350, 0.99390, 0.99310, 0.99400]
c60000 = [0.99200, 0.99350, 0.99500, 0.99440, 0.99450, 0.99410, 0.99400, 0.99350, 0.99410]
d60000 = [0.99190, 0.99380, 0.99390, 0.99380, 0.99460, 0.99360, 0.99290, 0.99380, 0.99350]

all60000 = [b60000, c60000, d60000]

plt.hlines(a60000, x60000[0], x60000[-1], label='Random sampling - baseline')
for i in range(3):
    plt.plot(x60000, all60000[i], label=sampling_methods[i])
plt.title('60000 samples')
plt.xlabel('Number of samples trained on')
plt.ylabel('Validation accuracy')
plt.legend()
plt.savefig('60000.png')

plt.clf()
