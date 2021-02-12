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
    scale = 180

    def __call__(self, image):
        return TF.affine(image, angle=0, translate=(0, 0), scale=1., 
                         shear=(self.sample_magnitude(), 0))


@transforms.register
class ShearY(Operation):
    scale = 180

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

    def __call__(self, image):
        return TF.rotate(image, self.sample_magnitude())


'''
@transforms.register
class AutoContrast(Operation):
'''


@transforms.register
class Invert(Operation):
    def __call__(self, image):
        if image.max() > 1 or image.min() < 0:
            raise ValueError('the value of image must lie between 0 and 1')

        return 1 - image


@transforms.register
class Equalize(Operation):
    # https://github.com/pytorch/vision/pull/3119/files
    def __call__(self, image):
        raise NotImplemented()


@transforms.register
class Solarize(Operation):
    # https://github.com/pytorch/vision/pull/3112/files
    def __call__(self, image):
        # assume given image is floats and values are between 0 and 1
        threshold = 1 - self.sample_magnitude().abs()
        mask = (image > threshold).float()
        image = mask * (1-image) + (1-mask) * image
        return image


@transforms.register
class Posterize(Operation):
    # https://github.com/pytorch/vision/pull/3108/files

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = int(1 + 7*self.magnitude)

    def __call__(self, image):
        if image.max() > 1 or image.min() < 0:
            raise ValueError('the value of image must lie between 0 and 1')

        image = (image * 255).int()
        mask = -int(2**(8 - bits))
        image = image & mask
        image = (image / 255.).float()
        return image


@transforms.register
class Contrast(Operation):
    bias = 1
    def __call__(self, image):
        return TF.adjust_contrast(image, self.sample_magnitude())


'''
@transforms.register
class Color(Operation):
'''


@transforms.register
class Brightness(Operation):
    bias = 1
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

