import torch
from PIL import Image

from network import SimpleNet
from utils import transforms

network = SimpleNet()
# load the network. In our case, the network is network_epoch_11.
network.load_state_dict(torch.load("models/network_epoch_11"))
# network.load_state_dict(torch.load("models/network_epoch_7"))
network.eval()
label_to_class = {1.0: 'face', 0.0: 'not_face'}


def test_image():
    while True:
        try:
            filename = input('Filename: ')
            file = f'test_images/{filename}'
            image = transforms(Image.open(file).convert('RGB'))
            image = image.unsqueeze(0)
            output = network(image)
            if output.data.numpy()[0] > 0.95:
                print("face")
            else:
                print("not face")
            print(output.data.numpy()[0])
        except Exception as e:
            print(e)


test_image()
