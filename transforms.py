import random
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF

PI = np.pi


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
    def __init__(self, magnitude):
        if not isinstance(magnitude, (list, tuple)):
            magnitude = [magnitude]
        self.magnitude = magnitude

    def __call__(self, image):
        raise NotImplemented()

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.magnitude}'

    def sample_magnitude(self, rand_negate=True):
        magnitude = random.uniform(min(self.magnitude), max(self.magnitude))
        assert 1 >= magnitude >= 0
        if not rand_negate:
            return magnitude
        return random.choice([-magnitude, magnitude])


transforms = BagOfOps()


@transforms.register
class ShearX(Operation):
    SCALE = 0.3 # 1. # max(Y/X) = 0.3

    def __call__(self, image):
        degree = self.sample_magnitude() * self.SCALE
        degree = np.arctan2(degree, 1) * 180 / PI
        return TF.affine(image, angle=0, translate=(0, 0), scale=1., 
                         shear=(degree, 0))


@transforms.register
class ShearY(Operation):
    SCALE = 0.3 # 1. # max(Y/X) = 0.3

    def __call__(self, image):
        degree = self.sample_magnitude() * self.SCALE
        degree = np.arctan2(degree, 1) * 180 / PI
        return TF.affine(image, angle=0, translate=(0, 0), scale=1., 
                         shear=(0, degree))


@transforms.register
class TranslateX(Operation):
    SCALE = 0.3125 # 0.3125 for cifar, 0.45 for imagenet

    def __call__(self, image):
        x_trans = image.size(-1) * self.SCALE # C, H, W
        return TF.affine(image, angle=0, 
                         translate=(x_trans*self.sample_magnitude(), 0), 
                         scale=1., shear=(0, 0))


@transforms.register
class TranslateY(Operation):
    SCALE = 0.3125 # 0.3125 for cifar, 0.45 for imagenet

    def __call__(self, image):
        y_trans = image.size(-2) * self.SCALE # C, H, W
        return TF.affine(image, angle=0, 
                         translate=(0, y_trans*self.sample_magnitude()), 
                         scale=1., shear=(0, 0))


@transforms.register
class Rotate(Operation):
    SCALE = 30 

    def __call__(self, image):
        return TF.rotate(image, self.sample_magnitude() * self.SCALE)


@transforms.register
class AutoContrast(Operation):
    # https://github.com/pytorch/vision/pull/3117/files
    def __call__(self, image):
        bound = 1.
        minimum = image.amin(dim=(-2, -1), keepdim=True)
        maximum = image.amax(dim=(-2, -1), keepdim=True)
        equal = torch.where(minimum == maximum)[0]
        minimum[equal] = 0
        maximum[equal] = bound
        scale = bound / (maximum - minimum)

        return ((image - minimum) * scale).clamp(0, bound)


# @transforms.register
class Invert(Operation):
    def __call__(self, image):
        return 1 - image


@transforms.register
class Equalize(Operation):
    # https://github.com/pytorch/vision/pull/3119/files
    def __call__(self, image):
        org_size = image.size()
        image = image.reshape(-1, *org_size[-2:])
        n_sample = image.size(0)

        image = (image * 255).long() # float to int

        for i in range(n_sample):
            # scale channel
            sample = image[i]
            hist = torch.histc(sample.float(), bins=256, min=0, max=255)
            nonzero_hist = hist[hist != 0]

            if nonzero_hist.numel() > 0:
                step = (nonzero_hist.sum() - nonzero_hist[-1]) // 255

                lut = (torch.cumsum(hist, 0) + (step // 2)) // step
                lut = torch.cat([torch.zeros(1, device=image.device), lut[:-1]])
                lut = lut.clamp(0, 255)
                image[i] = lut[sample]
            
        image = (image / 255.).float() # int to float 
        image = image.reshape(*org_size)
        image = image.clamp(0., 1.)
        return image


@transforms.register
class Solarize(Operation):
    def __call__(self, image):
        # assume given image is floats and values are between 0 and 1
        threshold = 1 - self.sample_magnitude(rand_negate=False)
        image = torch.where(image < threshold, image, 1-image)
        return image


@transforms.register
class Posterize(Operation):
    # https://github.com/pytorch/vision/pull/3108/files
    SCALE = 4 

    def __call__(self, image):
        if image.max() > 1 or image.min() < 0:
            raise ValueError('the value of image must lie between 0 and 1')

        image = (image * 255).int()
        bits = int(4 - self.sample_magnitude(rand_negate=False)*self.SCALE)
        mask = -int(2**(8 - bits))
        image = image & mask
        image = (image / 255.).float()
        image = image.clamp(0., 1.)
        return image


@transforms.register
class Contrast(Operation):
    def __call__(self, image):
        # [0.1, 1.9]
        magnitude = 0.1 + 1.8 * self.sample_magnitude(rand_negate=False)
        return TF.adjust_contrast(image, magnitude)


@transforms.register
class Color(Operation):
    def __call__(self, image):
        # [0.1, 1.9]
        color_balance = 0.1 + 1.8 * self.sample_magnitude(rand_negate=False)
        image = color_balance * image \
                + (1-color_balance) * TF.rgb_to_grayscale(image, 3)
        image = image.clamp(0., 1.)
        return image


@transforms.register
class Brightness(Operation):
    def __call__(self, image):
        # [0.1, 1.9]
        magnitude = 0.1 + 1.8 * self.sample_magnitude(rand_negate=False)
        return TF.adjust_brightness(image, magnitude)


# TODO: Fix this
@transforms.register
class Sharpness(Operation):
    # https://github.com/pytorch/vision/pull/3114/files
    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    def __call__(self, image):
        # [0.1, 1.9]
        sharpness = 0.1 + 1.8 * self.sample_magnitude(rand_negate=False)
        image = sharpness * image + (1-sharpness) * self.blur_image(image)
        image = image.clamp(0., 1.)
        return image

    @staticmethod
    def blur_image(image: torch.Tensor) -> torch.Tensor:
        image = image * 255

        n_chan = image.size(-3)
        kernel = torch.ones((3, 3), device=image.device)
        kernel[1, 1] = 5
        kernel /= kernel.sum()
        kernel = kernel.expand(n_chan, 1, 3, 3)

        single = image.ndim == 3
        if single:
            image = image.unsqueeze(0)

        degenerate = torch.nn.functional.conv2d(image, kernel, groups=n_chan)
        degenerate = degenerate.clamp(0., 255.)
        degenerate = degenerate.int()

        mask = torch.ones_like(degenerate) # [B, C, H-2, W-2]
        mask = torch.nn.functional.pad(mask, [1, 1, 1, 1, 0, 0, 0, 0])
        degenerate = torch.nn.functional.pad(degenerate, 
                                             [1, 1, 1, 1, 0, 0, 0, 0])
        image = image * (1-mask) + degenerate * mask

        if single:
            image = image.squeeze(0)

        image = (image / 255.).float()
        return image


# removed for RandAugment
# @transforms.register
class Cutout(Operation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_erasing = torchvision.transforms.RandomErasing(
            scale=(self.sample_magnitude(rand_negate=False)**2,)*2,
            ratio=(1, 1))

    def __call__(self, image):
        return self.random_erasing(image)


@transforms.register
class Identity(Operation):
    def __call__(self, image):
        return image


# @transforms.register
class SamplePairing(Operation):
    scale = 0.4

    def __call__(self, image):
        if image.ndim == 3: # single image:
            return image

        idx = torch.randperm(image.size(0))
        mag = self.sample_magnitude(rand_negate=False)
        return mag * image[idx] + (1-mag) * image


if __name__ == '__main__':
    print(transforms.ops)
    print(transforms.n_ops)

    x = torch.rand((3, 32, 32))
    xs = torch.rand((8, 3, 32, 32))

    import time
    for op in transforms.ops:
        o = op([0.1, 1])

        start = time.time()
        for i in range(100):
            assert(torch.isnan(o(x)).sum() == 0)
            output = o(xs)
            assert output.max() <= 1 and output.min() >= 0
        print(o, f'{time.time()-start:.3f}')

