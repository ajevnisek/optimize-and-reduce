import torch
import skimage
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from collections import Counter

from PIL import Image
from scipy.ndimage import median_filter
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans


def get_num_clusters(clustering_obj):
    return len(np.unique(clustering_obj.labels_))


def count_clusterless_pixels(labels_img):
    return (labels_img == -1).sum()


def bhattacharyya_distance(p,q):
    bc = np.sum(np.sqrt(p * q))
    return -np.log(bc)


class ClusteringMethod(Enum):
    DBSCAN = 1
    KMEANS = 2
    SPECTRAL_CLUSTERING = 3


class NonDifferentiableHistogramLoss:
    def __init__(self, base_image_tensor: torch.Tensor,
                 method: ClusteringMethod = ClusteringMethod.DBSCAN):
        self.base_image_tensor = base_image_tensor
        self.base_labels_image, self.base_hist, self.base_cluster_centers = \
            self.base_torch_image_to_clusters(base_image_tensor, method)


    @staticmethod
    def base_torch_image_to_clusters(normalized_tensor_image: torch.tensor,
                                     method: ClusteringMethod =
                                     ClusteringMethod.DBSCAN):
        resized_tensor = torch.nn.functional.interpolate(
            normalized_tensor_image.permute(2, 0, 1).unsqueeze(0),
            (100, 100)).squeeze(0).permute(1, 2, 0)
        numpy_image = resized_tensor.cpu().numpy() * 255.0
        # get sizes:
        (h, w) = numpy_image.shape[:-1]
        # prepare data:
        X = numpy_image[..., :3].reshape(h * w, -1)
        # cluster
        clustering_method = {
            ClusteringMethod.DBSCAN: DBSCAN(eps=5, min_samples=20, ),
            ClusteringMethod.KMEANS: KMeans(n_clusters=20),
            ClusteringMethod.SPECTRAL_CLUSTERING: SpectralClustering(
                n_clusters=20),
        }
        clustering = clustering_method[method].fit(X)
        # extract cluster centers:
        labels = clustering.labels_
        labels_image = labels.reshape(h, w)
        sorted_labels = np.unique(labels_image)
        sorted_labels.sort()

        cluster_centers = []
        for label in sorted_labels:
            indices = (clustering.labels_ == label)
            cluster_centers.append(
                np.array([
                    [X[indices, 0].mean(), X[indices, 1].mean(),
                     X[indices, 2].mean()]
                ])
            )

        if -1 in sorted_labels:
            cluster_centers = cluster_centers[1:] + [cluster_centers[0]]

        cluster_centers = np.concatenate(cluster_centers, axis=0)

        # handle unclustered labels
        if -1 in sorted_labels:
            kernel_size = 3

            while count_clusterless_pixels(labels_image) > 0:
                labels_filtered = median_filter(labels_image, size=kernel_size)
                new_labels_image = labels_image.copy()
                new_labels_image[labels_image == -1] = labels_filtered[
                    labels_image == -1]
                if count_clusterless_pixels(new_labels_image) >= \
                        count_clusterless_pixels(labels_image):
                    assert False, "Does not converge"
                labels_image = new_labels_image
                kernel_size += 2

        # create histogram:
        hist = Counter(labels_image.reshape(h * w))
        if -1 in sorted_labels:
            cluster_centers_to_return = cluster_centers[:-1] / 255.0
        else:
            cluster_centers_to_return = cluster_centers / 255.0
        return labels_image, hist, cluster_centers_to_return

    @staticmethod
    def show_histogram_of_cluster_centers(hist, cluster_centers, path=''):
        plt.clf()
        plt.bar(hist.keys(), hist.values(), color=cluster_centers)
        x_legend = ['(' + ', '.join([f'{int(x * 255)}' for x in list(cc)]) + ')'
                    for cc in cluster_centers]
        plt.xticks(np.arange(len(hist.keys())), x_legend, rotation=300)
        plt.title(
            'how many pixels per cluster center?\n count vs cluster center \n'
            f'total clusters: {len(hist.keys())}')
        plt.ylabel('count')
        plt.xlabel('cluster center')
        plt.yscale('log')
        plt.tight_layout()
        if path == '':
            plt.show()
        else:
            plt.savefig(path)

    def new_image_histogram(self, new_image: torch.Tensor):
        """

        """
        h, w = new_image.shape[0], new_image.shape[1]
        X = new_image.reshape(h * w, -1).cpu()

        L = pairwise_distances_argmin(X[..., :3], self.base_cluster_centers)
        # L = L.reshape(h, w)
        # colored_labels_new_image = cluster_centers[L]
        # plt.imshow(colored_labels_new_image);plt.show()
        # L = L.reshape(h * w)
        return Counter(L)

    def align_histogram_of_non_base(self, new_image_histogram: torch.Tensor):
        # put zeros in for cluster centers which do not exist in the new
        # image histogram:
        for cluster_center_index in self.base_hist:
            if cluster_center_index not in new_image_histogram:
                new_image_histogram[cluster_center_index] = 0
        return  new_image_histogram

    def rank(self, new_image: torch.Tensor):
        new_image_histogram = self.new_image_histogram(new_image)
        aligned_new_image_histogram = self.align_histogram_of_non_base(
            new_image_histogram)

        # convert histogram to normalized histograms
        q = np.array([self.base_hist[k] for k in sorted(self.base_hist.keys())])
        q = q / sum(q)
        p = np.array([aligned_new_image_histogram[k]
                      for k in sorted(aligned_new_image_histogram.keys())])
        p = p / sum(p)
        return bhattacharyya_distance(p, q)

    def __call__(self, base_image, other_image):
        return self.rank(other_image)


def read_image_from_path_to_normalized_tensor(path):
    numpy_image = skimage.io.imread(path)
    normalized_tensor_image = torch.from_numpy(numpy_image).to(
        torch.float32) / 255.0
    return normalized_tensor_image


def main():
    path = 'target_images/apes/23126082.png'
    # path = 'target_images/tiger.png'
    normalized_tensor_image = read_image_from_path_to_normalized_tensor(path)
    hist_loss = NonDifferentiableHistogramLoss(normalized_tensor_image,
                                               method=ClusteringMethod.DBSCAN)

    colored_labels = hist_loss.base_cluster_centers[hist_loss.base_labels_image
                     ] * 255.0
    plt.subplot(1, 2, 1)
    plt.imshow(normalized_tensor_image)
    plt.title('image')
    plt.subplot(1, 2, 2)
    plt.imshow(colored_labels.astype(np.uint8))
    plt.title('labels')
    plt.show()
    hist_loss.show_histogram_of_cluster_centers(hist_loss.base_hist,
                                      hist_loss.base_cluster_centers)
    images_to_path = {
        'some noisy tiger': 'results/diffvg_tiger_l1_and_clip_mix_alpha_0/'
                            'result.png',
        'an ape': 'target_images/apes/23126082.png'}
    import ipdb; ipdb.set_trace()
    for description in images_to_path:
        new_image_path = images_to_path[description]
        new_normalized_tensor_image = read_image_from_path_to_normalized_tensor(
            new_image_path)
        new_histogram = hist_loss.new_image_histogram(
            new_normalized_tensor_image)
        aligned_new_histogram = hist_loss.align_histogram_of_non_base(
            new_histogram)
        hist_loss.show_histogram_of_cluster_centers(
            aligned_new_histogram, hist_loss.base_cluster_centers)
        distance = hist_loss.rank(new_normalized_tensor_image)
        print(f"Bhattacharyya distance to {description}: "
              f" {distance:.2f}")


if __name__ == "__main__":
    main()
