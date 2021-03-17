import torch
import torchvision


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


x = torch.rand(4, 3, 32, 32)
model = torchvision.models.resnet18(pretrained=False)
avg = model.avgpool.register_forward_hook(get_activation('avgpool'))

out = model(x)
print(out)
print(model.fc(activation['avgpool'].squeeze()))

avg.remove() # detach hooks
'''

