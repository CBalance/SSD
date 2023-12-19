import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image


def visualize_output_and_save(imgid, output,config,gt, save_path):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """
    file_path = config.DATA.DATA_PATH + '/images_384x576/' + imgid + '.jpg'
    origin_img = Image.open(file_path).convert("RGB")
    origin_img = np.array(origin_img)
    h, w, _ = origin_img.shape

    cmap = plt.cm.get_cmap('jet')
    density_map = output
    pred = output.sum().item()
    density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(
        0).numpy()

    density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0
    density_map = density_map[:, :, 0:3] * 0.5 + origin_img * 0.5

    fig = plt.figure(dpi=800)
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax2.set_title(str(pred) + "  " + str(gt))
    ax2.imshow(density_map.astype(np.uint8))

    plt.savefig(save_path)
    plt.close()



