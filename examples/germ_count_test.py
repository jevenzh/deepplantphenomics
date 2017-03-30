from deepplantphenomics import networks
import os

dir = './data/germ'

images = [os.path.join(dir, name) for name in os.listdir(dir) if
          os.path.isfile(os.path.join(dir, name)) & name.endswith('.png')]

net = networks.germRegressor(batch_size=4)
y = net.forward_pass(images)
net.shut_down()

for k,v in zip(images, y):
    print '%s: %d' % (os.path.basename(k), v)

print('Done')