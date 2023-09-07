import torch
import matplotlib.pyplot as plt

class PatchCollate:

    def __init__(self, kernel, stride):
        self.kernel, self.stride = kernel, stride

    def __call__(self, x):
        x, labels = torch.utils.data.default_collate(x)
        b, c, _, _ = x.shape
        windows = x.unfold(2, self.kernel, self.stride).unfold(3, self.kernel, self.stride).permute(0, 2, 3, 1, 4, 5)
        return windows.reshape(b, -1, c, windows.shape[-2], windows.shape[-1]), labels
    

class ToRGB:

    def __init__(self):
        pass

    def __call__(self, img):
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        return img
    

def test_image(loader, idx_map):
    imgs, labels = next(iter(loader))
    output, output_label = imgs[0], idx_map[labels[0].item()]
    fig, ax = plt.subplots(nrows=16, ncols=16, figsize=(20, 10))
    ax = ax.flatten()
    for patch, axis in zip(output, ax):
        axis.imshow(patch.permute(1, 2, 0))
        axis.axis(False)
    fig.suptitle(output_label)
    plt.show()