import numpy as np
import pandas as pd


class Statistics:
    def __init__(self) -> None:
        pass

    def per_instance_stats(self, tree, image):
        """Gets some statistics for a specific tree/canopy

        Args:
            mask (np.array(int)): Mask as produced by split_mask_*, either the trees or canopies
            image (np.array(int)): Image
            index_instance (int): The integer for the tree/canopy for which the stats have to be created

        Returns:
            dict: Dictionary with some stats
        """
        image_values = tree.get_pixels(image)
        instance_stats = {
            "x": tree.polygon.centroid[0][0],
            "y": tree.polygon.centroid[0][1],
            "pixel_size": tree.polygon.area(),
            "red_value": np.mean(image_values[:, 0]),
            "green_value": np.mean(image_values[:, 1]),
            "blue_value": np.mean(image_values[:, 2]),
        }
        return instance_stats

    def run(self, processed_result):

        general_stats = {
            "n_trees": len(processed_result.trees),
            "tree_cover": np.count_nonzero(processed_result.tree_mask)
            / np.prod(processed_result.image.shape[:2]),
            "canopy_cover": np.count_nonzero(processed_result.canopy_mask)
            / np.prod(processed_result.image.shape[:2]),
            "tree_canopy_cover": np.count_nonzero(
                np.logical_or(processed_result.tree_mask, processed_result.canopy_mask)
            )
            / np.prod(processed_result.image.shape[:2]),
        }

        tree_stats = []
        for tree in np.unique(processed_result.trees):
            tree_stats.append(self.per_instance_stats(tree, processed_result.image))

        return general_stats, pd.DataFrame(tree_stats)
