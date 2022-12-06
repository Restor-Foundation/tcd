import numpy as np
import pandas as pd


class Statistics:
    def __init__(self) -> None:
        pass

    def _per_instance_stats(self, tree, image):
        """
        Gets some statistics for a specific tree/canopy

        Args:
            tree (ProcessedInstance): a tree
            image (np.array(int)): Image

        Returns:
            dict: Dictionary with some stats
        """
        image_values = tree.get_pixels(image)
        instance_stats = {
            "x": tree.polygon.centroid.coords[0][0],
            "y": tree.polygon.centroid.coords[0][1],
            "pixel_size": tree.polygon.area,
            "red_value": np.mean(image_values[:, 0]),
            "green_value": np.mean(image_values[:, 1]),
            "blue_value": np.mean(image_values[:, 2]),
        }
        return instance_stats

    def run(self, processed_result):
        """Runs the result and gets some statistics in general and for each tree

        Args:
            processed_result (ProcessedResult): a processed result

        Returns:
            dict, pd.DataFrame: Dict contains some general statistics, DataFrame contains statistics per tree
        """
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
        for tree in processed_result.trees:
            tree_stats.append(self._per_instance_stats(tree, processed_result.image))

        return general_stats, pd.DataFrame(tree_stats)
