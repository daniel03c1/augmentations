import torch
import torchvision.transforms.functional as TF

# https://github.com/pytorch/vision/issues/3050
# automatically registers operations (use @transforms.register)
class BagOfOps:
    def __init__(self):
        self.ops = []
        self.n_ops = 0

    def register(self, op):
        if op not in self.ops:
            self.ops.append(op)
            self.n_ops += 1
        return op

    def __getitem__(self, index):
        return self.ops[index]


class Operation:
    scale = 1
    bias = 0

    def __init__(self, magnitude):
        assert 1 >= magnitude >= 0
        self.magnitude = magnitude
        self.range = [-magnitude*self.scale + self.bias,
                       magnitude*self.scale + self.bias]

    def __call__(self, image):
        raise NotImplemented()

    def sample_magnitude(self):
        return torch.empty(1).uniform_(*self.range).item()

    def __repr__(self):
        return f'{self.__class__.__name__}: [{self.range[0]}, {self.range[1]}]'


transforms = BagOfOps()


@transforms.register
class ShearX(Operation):
    def __call__(self, image):
        return TF.affine(image, angle=0, translate=(0, 0), scale=1., 
                         shear=(self.sample_magnitude(), 0))


@transforms.register
class ShearY(Operation):
    def __call__(self, image):
        return TF.affine(image, angle=0, translate=(0, 0), scale=1., 
                         shear=(0, self.sample_magnitude()))


@transforms.register
class TranslateX(Operation):
    def __call__(self, image):
        return TF.affine(image, angle=0, 
                         translate=(self.sample_magnitude(), 0), 
                         scale=1., shear=(0, 0))


@transforms.register
class TranslateY(Operation):
    def __call__(self, image):
        return TF.affine(image, angle=0, 
                         translate=(0, self.sample_magnitude()), 
                         scale=1., shear=(0, 0))


@transforms.register
class Rotate(Operation):
    scale = 180
    bias = 0

    def __call__(self, image):
        return TF.rotate(image, self.sample_magnitude())


'''
@transforms.register
class AutoContrast(Operation):
'''


@transforms.register
class Invert(Operation):
    def __call__(self, image):
        bound = 1.0 if image.is_floating_point() else 255.0
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        return (bound - img.to(dtype)).to(img.dtype)


'''
@transforms.register
class Equalize(Operation):


@transforms.register
class Solarize(Operation):


@transforms.register
class Posterize(Operation):
'''


@transforms.register
class Contrast(Operation):
    def __call__(self, image):
        return TF.adjust_contrast(image, self.sample_magnitude())


'''
@transforms.register
class Color(Operation):
'''


@transforms.register
class Brightness(Operation):
    def __call__(self, image):
        return TF.adjust_brightness(image, self.sample_magnitude())


'''
@transforms.register
class Sharpness(Operation):


@transforms.register
class Cutout(Operation):


@transforms.register
class SamplePairing(Operation):
'''


if __name__ == '__main__':
    print(transforms.ops)
    print(transforms.n_ops)

    x = torch.zeros((32, 32, 3))
    x = Rotate(30/180)(x)
    print(Rotate(30/180))
    print(transforms[0](0.1))

