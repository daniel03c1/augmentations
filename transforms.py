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
    def __init__(self, magnitude):
        self.magnitude

    def __call__(self, image):
        pass


transforms = BagOfOps()


@transforms.register
class ShearX(Operation):
    def __call__(self, image):
        return image


@transforms.register
class ShearY(Operation):
    def __call__(self, image):
        return image


@transforms.register
class TranslateX(Operation):
    def __call__(self, image):
        return image


@transforms.register
class TranslateY(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Rotate(Operation):
    def __call__(self, image):
        return image


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
        return image


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

