import os

#create resnet
with open('./run.sh', 'w') as f:
    f.write('python main.py --optimizer SGD --model resnet --depth 20 --index 0 &\n')

#create resnet for SGD_layer
for i in range(1, 5):
    with open('./run.sh', 'a') as f:
        f.write('python main.py --optimizer SGD_layer --model resnet --depth 20 --index {} --lr 1e-{} & \n'.format(i, i-1))

#create resnet for SGD_spectral
for i in range(6, 16):
    with open('./run.sh', 'a') as f:
        f.write('python main.py --optimizer SGD_spectral --model resnet --depth 20 --index {} --lam 1e-{} & \n'.format(i, i-4))


#create vgg
with open('./run.sh', 'a') as f:
    f.write('python main.py --optimizer SGD --model vgg --depth 11 --index 16\n')

for i in range(1, 5):
    with open('./run.sh', 'a') as f:
        f.write('python main.py --optimizer SGD_layer --model vgg --depth 11 --index {} --lr 1e-{} & \n'.format(i+16, i-1))

for i in range(6, 16):
    with open('./run.sh', 'a') as f:
        f.write('python main.py --optimizer SGD_spectral --model vgg --depth 11 --index {} --lam 1e-{} & \n'.format(i+16, i-4))

