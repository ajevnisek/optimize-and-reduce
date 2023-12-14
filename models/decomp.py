import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from sklearn.cluster import KMeans
import torch
import random as rng
import numpy.random as npr
import copy
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
import networkx as nx
from tqdm import tqdm


def maskLoss(recon, trgt, masks):
    loss = 0
    for i, mask in enumerate(masks):
        h, w = mask.shape
        masked_recon = recon[i][3] * recon[i][:3] + (1 - recon[i][3]) * torch.ones_like(recon[i][:3])
        masked_trgt = mask * trgt + (1 - mask) * torch.ones_like(trgt)
        loss += ((masked_recon - masked_trgt) ** 2).mean() / mask.sum() * h * w
    return loss


def create_masks(img, fg_mask, num_shapes=4, num_cluster_centers=20, ignore_clusters_smaller_than=0.01, sigma_const=0.001, device='cpu', output_dir=''):
    img = np.transpose(img.squeeze().cpu().numpy(), (1, 2, 0))
    H, W, C = img.shape
    fg_mask = cv2.resize(fg_mask, dsize=(H, W), interpolation=cv2.INTER_CUBIC)
    X = img.reshape(H * W, C)
    cluster = KMeans(num_cluster_centers).fit(X)
    color_clusters = cluster.labels_.reshape(H, W)
    cluster_indices = np.unique(color_clusters)
    relevant_cluster_indices = [
        c for c in cluster_indices
        if (color_clusters == c).sum() > H * W * ignore_clusters_smaller_than]
    if len(relevant_cluster_indices) != num_cluster_centers:
        print(f'Narrowed down cluster number from: {num_cluster_centers} to:'
              f'{len(relevant_cluster_indices)} clusters.')
    new_K = len(relevant_cluster_indices)
    cluster = KMeans(new_K).fit(X)
    map = cluster.labels_.reshape(H, W)
    idcnt = {}
    for idi in range(new_K):
        idcnt[idi] = (((map == idi) * fg_mask).sum(), cluster.cluster_centers_[idi])
    counter = 0
    shapes = []
    for i, (_, color) in tqdm(sorted(idcnt.items(), key=lambda item: item[1][0], reverse=True)):
        sigmas = sigma_const * idcnt[i][0]
        mask = (map == i).astype(np.float32)
        downsampled_mask = cv2.resize(mask, dsize=(128, 128), interpolation=cv2.INTER_CUBIC) * \
                           cv2.resize(fg_mask, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        blurred_mask = gaussian_filter(downsampled_mask, sigma=sigmas)
        # blurred_mask = np.pad(blurred_mask, [(pad_width, pad_width), (pad_width, pad_width)], mode='constant')
        mrf_mask = run_mrf_on_graph(blurred_mask)
        _, component, cstats, ccenter = cv2.connectedComponentsWithStats(mrf_mask.astype(np.uint8), connectivity=4)
        for j in np.unique(component):
            if j != 0:
                comp_mask = cv2.resize((component == j).astype(np.uint8), dsize=(H, W), interpolation=cv2.INTER_CUBIC)
                mean_row, mean_col = [int(x.mean()) for x in np.where(comp_mask > 0.5)]
                shapes.append((comp_mask, (component == j).sum(), counter, color, (mean_row, mean_col)))
        counter += 1

    shapes_by_size = sorted(shapes, key=lambda shape: shape[1], reverse=True)
    if len(shapes) > num_shapes:
        shapes_by_size = shapes_by_size[:num_shapes]
        # shapes_by_layer = sorted(shapes_by_size, key=lambda shape: shape[2])

    result = np.ones((H, W, 3))
    for mask in shapes_by_size:
        result[mask[0] > 0.5, ...] = mask[3]

    cv2.imwrite("{}/{}".format(output_dir, "masks.png"), (result*255).astype(np.uint8)[..., ::-1])

    masks = [shape[0] for shape in shapes_by_size]
    coords = [shape[-1] for shape in shapes_by_size]

    trimmed_object_masks_tensor = [torch.tensor(x).to(device) for x in masks]
    coords_tensor = [torch.cat([torch.tensor(x).to(device)[None], torch.tensor(y).to(device)[None]]) for (x, y) in coords]
    return trimmed_object_masks_tensor, coords_tensor


def run_mrf_on_graph(blurred_mask, binary_term_weight=0.5):
    H, W = blurred_mask.shape
    unary_term_weights_connected_to_source = blurred_mask
    unary_term_weights_connected_to_sink = 1 - blurred_mask
    G = nx.DiGraph()
    for h in range(H):
        for w in range(W):
            for i in [1]:
                for j in [1]:
                    if H > h + i > 0 and W > w + j > 0:
                        G.add_edge(str((h, w)), str((h + i, w)),
                                   capacity=np.absolute(
                                       blurred_mask[h + i, w] - blurred_mask[h, w]) * binary_term_weight)
                        G.add_edge(str((h, w)), str((h, w + j)),
                                   capacity=np.absolute(
                                       blurred_mask[h, w + j] - blurred_mask[h, w]) * binary_term_weight)
            G.add_edge("SOURCE", str((h, w)),
                       capacity=unary_term_weights_connected_to_source[h, w])

            G.add_edge(str((h, w)), "SINK",
                       capacity=unary_term_weights_connected_to_sink[h, w])
    cut_value, partition = nx.minimum_cut(G, "SOURCE", "SINK")
    mask = np.zeros((H, W))
    for item in partition[0]:
        if item != "SOURCE":
            i, j = item[1:-1].split(",")
            mask[int(i), int(j)] = 1
    return mask


class Sparse_coord_init:
    def __init__(self, pred, gt, format='[2D x c]', num_cluster_centers=10, ignore_clusters_smaller_than=0.5 / 100.,
                 quantile_interval=20, nodiff_thres=0.1):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0]) ** 2).sum(0)
            self.reference_gt = copy.deepcopy(
                np.transpose(gt[0], (1, 2, 0)))
        elif format == '[2D x c]':
            self.map = (np.abs(pred - gt)).sum(-1)
            self.reference_gt = copy.deepcopy(gt[0])
        else:
            raise ValueError
        self.num_cluster_centers = num_cluster_centers
        # OptionA: Zero too small errors to avoid the error too small deadloop
        H, W, C = gt.shape
        X = gt.reshape(H * W, C)
        cluster = KMeans(self.num_cluster_centers).fit(X)
        color_clusters = cluster.labels_.reshape(H, W)
        cluster_indices = np.unique(color_clusters)
        relevant_cluster_indices = [
            c for c in cluster_indices
            if (color_clusters == c).sum() > H * W * ignore_clusters_smaller_than]
        if len(relevant_cluster_indices) != self.num_cluster_centers:
            print(f'Narrowed down cluster number from: {self.num_cluster_centers} to:'
                  f'{len(relevant_cluster_indices)} clusters.')
        cluster = KMeans(len(relevant_cluster_indices)).fit(X)
        self.map = cluster.labels_.reshape(H, W)
        self.idcnt = {}
        for idi in sorted(np.unique(self.map)):
            self.idcnt[idi] = (self.map == idi).sum()
        self.idcnt.pop(min(self.idcnt.keys()))
        # remove smallest one to remove the correct region

    def __call__(self):
        if len(self.idcnt) == 0:
            h, w = self.map.shape
            return [npr.uniform(0, 1) * w, npr.uniform(0, 1) * h]
        target_id = max(self.idcnt, key=self.idcnt.get)
        _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
            (self.map == target_id).astype(np.uint8), connectivity=4)

        # remove cid = 0, it is the invalid area
        csize = [ci[-1] for ci in cstats[1:]]
        target_cid = csize.index(max(csize)) + 1
        center = ccenter[target_cid][::-1]
        coord = np.stack(np.where(component == target_cid)).T
        dist = np.linalg.norm(coord - center, axis=1)
        target_coord_id = np.argmin(dist)
        coord_h, coord_w = coord[target_coord_id]
        # replace_sampling
        self.idcnt[target_id] -= max(csize)
        if self.idcnt[target_id] == 0:
            self.idcnt.pop(target_id)
        self.map[component == target_cid] = 0
        return (component == target_cid).astype(np.uint8), coord_w, coord_h


class Decomp:
    def __init__(self, num_cluster_centers=8,
                 ignore_clusters_smaller_than=0.1 / 100.0,
                 ignore_shapes_smaller_than=0.1 / 100.,
                 add_positional_encoding=False,
                 device='cpu'):
        super(Decomp, self).__init__()
        self.add_positional_encoding = add_positional_encoding
        self.num_cluster_centers = num_cluster_centers
        self.ignore_clusters_smaller_than = ignore_clusters_smaller_than
        self.ignore_shapes_smaller_than = ignore_shapes_smaller_than
        self.device = device

    def decomp(self, img):
        img = np.transpose(img.squeeze().cpu().numpy(), (1, 2, 0))
        H, W, C = img.shape
        masks, coords = [], []
        sparse_coord_init = Sparse_coord_init(np.ones_like(img), img)
        for shape in range(self.num_cluster_centers):
            mask, coord_w, coord_h = sparse_coord_init()
            masks.append(mask)
            coords.append((coord_h, coord_w))

        trimmed_object_masks_tensor = [torch.tensor(x).to(self.device) for x in masks]
        coords_tensor = [torch.cat([torch.tensor(x).to(self.device)[None], torch.tensor(y).to(self.device)[None]])
                         for (x, y) in coords]
        return trimmed_object_masks_tensor, coords_tensor

        if self.add_positional_encoding:
            cols, rows = np.meshgrid(np.arange(W), np.arange(H))
            img = np.concatenate([img, cols[..., np.newaxis], rows[..., np.newaxis]],
                                 axis=2)
            C += 2

        X = img.reshape(H * W, C)
        cluster = KMeans(self.num_cluster_centers).fit(X)
        color_clusters = cluster.labels_.reshape(H, W)

        # plt.subplot(1, 2, 1)
        # plt.title('input image')
        # plt.imshow(Image.open(img_path))
        # plt.subplot(1, 2, 2)
        # plt.title('color clusters')
        # plt.imshow(color_clusters, cmap=plt.colormaps.get('Pastel1'))
        # plt.colorbar()
        # fig = plt.gcf()
        # fig.set_size_inches((8, 8))
        # plt.savefig(os.path.join(RESULTS_DIR, f'color_clusters_for_{image_name}.png'))

        cluster_indices = np.unique(color_clusters)
        relevant_cluster_indices = [
            c for c in cluster_indices
            if (color_clusters == c).sum() > H * W * self.ignore_clusters_smaller_than]

        if len(relevant_cluster_indices) != self.num_cluster_centers:
            print(f'Narrowed down cluster number from: {self.num_cluster_centers} to:'
                  f'{len(relevant_cluster_indices)} clusters.')

        print(f"Creating connected components from clusters...")
        object_masks = []
        for cluster_index in relevant_cluster_indices:

            # canny_output = cv2.Canny((color_clusters == cluster_index).astype(np.uint8), 0, 1)
            # contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # # Draw contours
            # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            # contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # for i in range(min(4, len(contours))):
            #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            #     cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

            num_labels, components_image = cv2.connectedComponents(
                (color_clusters == cluster_index).astype(np.uint8))
            for label in range(num_labels):
                cluster_histogram_in_curr_component = Counter(
                    color_clusters[components_image == label])
                if cluster_index not in cluster_histogram_in_curr_component:
                    continue
                most_common = cluster_histogram_in_curr_component.most_common(1)[0][0]
                if cluster_index != most_common:
                    continue
                object_masks.append((components_image == label).astype(np.uint8))

        print(f"Sorting connected components according to area...")
        object_masks = sorted(object_masks, key=lambda x: x.sum(), reverse=True)

        print(f"Ignoring connected components with area < "
              f"{self.ignore_shapes_smaller_than * 100.0:.2f} of image size...")
        trimmed_object_masks = [mask for mask in object_masks
                                if mask.sum() > H * W * self.ignore_shapes_smaller_than]
        trimmed_object_masks_tensor = [torch.tensor(x).to(self.device) for x in trimmed_object_masks]
        return trimmed_object_masks_tensor

        # print(f"Plotting clusters and connected components.")
        # from math import floor, sqrt
        # plot_rows = 1 + int(floor(sqrt(len(object_masks))))
        # plot_rows = 3
        # original_image = Image.open(img_path)
        #
        # plt.subplot(3, 2, 1)
        # plt.title('input image')
        # plt.imshow(original_image)
        # plt.subplot(3, 2, 2)
        # plt.title('color clusters')
        # plt.imshow(color_clusters, cmap=plt.colormaps.get('Pastel1'))
        # plt.colorbar()
        #
        # for i in range(6):
        #     if i < len(object_masks):
        #         mask = object_masks[i]
        #         plt.subplot(plot_rows, plot_rows, plot_rows + i+1)
        #         plt.imshow(object_masks[i], cmap=plt.colormaps.get('Pastel1'))
        #
        # fig = plt.gcf()
        # fig.set_size_inches((8, 8))
        # plt.savefig(os.path.join(RESULTS_DIR,
        #                          f'connected_components_for_{image_name}.png'))
