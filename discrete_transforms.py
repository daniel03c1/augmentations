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
    def __init__(self, probs: torch.Tensor, min_mag=0, max_mag=1):
        self.probs = probs
        self.div = len(probs) - 1
        self.min_mag = min_mag
        self.max_mag = max_mag

    def __call__(self, image):
        raise NotImplemented()

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.min_mag}, {self.max_mag}'

    def _sample_mag(self):
        magnitude = self.probs.multinomial(1)[0].item() / self.div
        return self.min_mag + magnitude * (self.max_mag - self.min_mag)


transforms = BagOfOps()


@transforms.register
class ShearX(Operation):
    def __init__(self, probs, min_mag=-1, max_mag=1):
        super(ShearX, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        degree = self._sample_mag()
        degree = np.arctan2(degree, 1) * 180 / PI
        return TF.affine(image, angle=0, translate=(0, 0), scale=1., 
                         shear=(degree, 0))


@transforms.register
class ShearY(Operation):
    def __init__(self, probs, min_mag=-1, max_mag=1):
        super(ShearY, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        degree = self._sample_mag()
        degree = np.arctan2(degree, 1) * 180 / PI
        return TF.affine(image, angle=0, translate=(0, 0), scale=1., 
                         shear=(0, degree))


@transforms.register
class TranslateX(Operation):
    def __init__(self, probs, min_mag=-0.5, max_mag=0.5):
        super(TranslateX, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        x_trans = image.size(-1) * self._sample_mag()
        return TF.affine(image, angle=0, translate=(x_trans, 0), 
                         scale=1., shear=(0, 0))


@transforms.register
class TranslateY(Operation):
    def __init__(self, probs, min_mag=-0.5, max_mag=0.5):
        super(TranslateY, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        x_trans = image.size(-1) * self._sample_mag()
        return TF.affine(image, angle=0, translate=(x_trans, 0), 
                         scale=1., shear=(0, 0))


@transforms.register
class Rotate(Operation):
    def __init__(self, probs, min_mag=-180, max_mag=180):
        super(Rotate, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        return TF.rotate(image, self._sample_mag())


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


# @transforms.register
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
        threshold = 1 - self._sample_mag()
        image = torch.where(image < threshold, image, 1-image)
        return image


@transforms.register
class Posterize(Operation):
    def __init__(self, probs, min_mag=0, max_mag=8):
        super(Posterize, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        if image.max() > 1 or image.min() < 0:
            raise ValueError('the value of image must lie between 0 and 1')

        image = (image * 255).int()
        bits = int(self._sample_mag())
        mask = -int(2**(8 - bits))
        image = image & mask
        image = (image / 255.).float()
        image = image.clamp(0., 1.)
        return image


@transforms.register
class Contrast(Operation):
    def __init__(self, probs, min_mag=0.1, max_mag=1.9):
        super(Contrast, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        return TF.adjust_contrast(image, self._sample_mag())


@transforms.register
class Color(Operation):
    def __init__(self, probs, min_mag=0.1, max_mag=1.9):
        super(Color, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        color_balance = self._sample_mag()
        image = color_balance * image \
                + (1-color_balance) * TF.rgb_to_grayscale(image, 3)
        image = image.clamp(0., 1.)
        return image


@transforms.register
class Brightness(Operation):
    def __init__(self, probs, min_mag=0.1, max_mag=1.9):
        super(Brightness, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        # [0.1, 1.9]
        return TF.adjust_brightness(image, self._sample_mag())


@transforms.register
class Sharpness(Operation):
    def __init__(self, probs, min_mag=0.1, max_mag=1.9):
        super(Sharpness, self).__init__(probs, min_mag, max_mag)

    def __call__(self, image):
        sharpness = self._sample_mag()
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


@transforms.register
class Cutout(Operation):
    def __init__(self, probs, min_mag=0, max_mag=1):
        super(Cutout, self).__init__(probs, min_mag, max_mag)
        self.random_erasing = torchvision.transforms.RandomErasing(
            scale=(0, 0), ratio=(1, 1))

    def __call__(self, image):
        self.random_erasing.scale = (self._sample_mag()**2,) * 2
        return self.random_erasing(image)


@transforms.register
class Identity(Operation):
    def __call__(self, image):
        return image


@transforms.register
class Zeros(Operation):
    def __call__(self, image):
        return image * 0


@transforms.register
class Danger(Operation):
    def __call__(self, image):
        if self._sample_mag() < 0.75:
            return image * 0
        return image


if __name__ == '__main__':
    print(transforms.ops)
    print(transforms.n_ops)

    x = torch.rand((3, 32, 32))
    xs = torch.rand((8, 3, 32, 32))

    import time
    for op in transforms.ops:
        probs = torch.rand(16)
        probs /= probs.sum()
        o = op(probs)

        start = time.time()
        for i in range(100):
            assert(torch.isnan(o(x)).sum() == 0)
            output = o(xs)
            assert output.max() <= 1 and output.min() >= 0
        print(o, f'{time.time()-start:.3f}')

