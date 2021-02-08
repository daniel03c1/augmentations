import torch
import torchvision.transforms.functional as TF


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


class Operation:
    def __init__(self, magnitude_range):
        if isinstance(magnitude_range, (int, float)):
            magnitude_range = [-magnitude_range, magnitude_range]
        assert len(magnitude_range) == 2, 'range must be a pair of numbers(float)'
        self.range = magnitude_range

    def __call__(self, image):
        pass

    def sample_magnitude(self):
        return torch.empty(1).uniform_(*self.range).item()


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
    def __call__(self, image):
        return TF.rotate(image, self.sample_magnitude())


@transforms.register
class AutoContrast(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Invert(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Equalize(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Solarize(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Posterize(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Contrast(Operation):
    def __call__(self, image):
        return TF.adjust_contrast(image, self.sample_magnitude())


@transforms.register
class Color(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Brightness(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Sharpness(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Cutout(Operation):
    def __call__(self, image):
        return image


@transforms.register
class SamplePairing(Operation):
    def __call__(self, image):
        return image


if __name__ == '__main__':
    print(transforms.ops)
    print(transforms.n_ops)

    x = torch.zeros((32, 32, 3))
    x = Rotate(30)(x)

