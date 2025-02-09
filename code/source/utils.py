import torch
import os
import numpy as np
import torch.nn.functional as F

PLOTCOLORS = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "pink": "#f781bf",
    "brown": "#a65628",
    "purple": "#984ea3",
    "gray": "#999999",
    "red": "#e41a1c",
    "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90",
    "yellow": "#dede00",
}


def ConvSingularValues(kernel, input_shape):
    transforms = torch.fft.fft2(kernel.permute(2, 3, 0, 1), input_shape, dim=[0, 1])
    print(transforms.shape)
    return torch.linalg.svd(transforms)


def ConvSingularValuesNumpy(kernel, input_shape):
    kernel = kernel.detach().cpu().numpy()
    transforms = np.fft.fft2(kernel.transpose(2, 3, 0, 1), input_shape, axes=[0, 1])
    # transforms = np.fft.fft2(kernel.permute(2,3,0,1), input_shape, dim=[0,1])
    print(transforms.shape)
    return np.linalg.svd(transforms)


def EigenValues(kernel, input_shape):
    # transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    transforms = torch.fft.fft2(kernel.permute(2, 3, 0, 1), input_shape, dim=[0, 1])
    print(transforms.shape)
    return torch.linalg.eig(transforms)


def load_state_dict_ignore_size_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    matched_state_dict = {}

    for key, param in state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == param.shape:
                matched_state_dict[key] = param
            else:
                print(
                    f"Size mismatch for key '{key}': model {model_state_dict[key].shape}, checkpoint {param.shape}"
                )
        else:
            print(f"Key '{key}' not found in model state dict.")

    model_state_dict.update(matched_state_dict)
    model.load_state_dict(model_state_dict)


def compare_optimizer_state_dicts(original, modified):
    diff = {}
    for key in original.keys():
        if key not in modified:
            diff[key] = "Removed"
        elif original[key] != modified[key]:
            diff[key] = {"Original": original[key], "Modified": modified[key]}
    for key in modified.keys():
        if key not in original:
            diff[key] = "Added"
    return diff


def get_worker_init_fn(start, end):
    return lambda worker_id: os.sched_setaffinity(0, range(start, end))


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ["0", "n", "f"]:
        return False
    elif x[0] in ["1", "y", "t"]:
        return True
    raise ValueError("Invalid value: {}".format(x))


def apply_pca(x, n_components=3):
    # x.shape = [B, C, H, W]
    from sklearn.decomposition import PCA

    pca = PCA(n_components)
    nx = []
    d = x.shape[1]
    for _x in x:
        _x = _x.permute(1, 2, 0).reshape(-1, d)
        _x = pca.fit_transform(_x)
        _x = _x.transpose(1, 0).reshape(n_components, x.shape[2], x.shape[3])
        nx.append(torch.tensor(_x))
    nx = torch.stack(nx, 0)
    # normalize to [0, 1]
    nx = (nx - nx.min()) / (nx.max() - nx.min())
    return nx


def apply_pca_torch(x, n_components=3):
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    N = H * W

    if n_components >= C:
        return x

    # Reshape to [B, N, C]
    x = x.permute(0, 2, 3, 1).reshape(B, N, C)

    # Center the data per sample
    x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, C]
    x_centered = x - x_mean  # [B, N, C]

    # Compute covariance matrix per sample: [B, C, C]
    cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (N - 1)

    # Compute eigenvalues and eigenvectors per sample
    eigenvalues, eigenvectors = torch.linalg.eigh(
        cov
    )  # eigenvalues: [B, C], eigenvectors: [B, C, C]

    # Reverse the order of eigenvalues and eigenvectors to get descending order
    eigenvalues = eigenvalues.flip(dims=[1])
    eigenvectors = eigenvectors.flip(dims=[2])

    # Select the top 'dim' eigenvectors
    top_eigenvectors = eigenvectors[:, :, :n_components]  # [B, C, dim]

    # Project the centered data onto the top eigenvectors
    x_pca = torch.bmm(x_centered, top_eigenvectors)  # [B, N, dim]

    # Reshape back to [B, dim, H, W]
    x_pca = x_pca.transpose(1, 2).reshape(B, n_components, H, W)

    return x_pca


def gen_saccade_imgs(img, psize, r):
    H, W = img.shape[-2:]
    img = F.interpolate(img, (H + psize - r, W + psize - r), mode="bicubic")
    imgs = []
    for h in range(0, psize, r):
        for w in range(0, psize, r):
            imgs.append(img[:, :, h : h + H, w : w + W])
    return imgs, img[:, :, psize // 2 : H + psize // 2, psize // 2 : W + psize // 2]
