import numpy as np
import pandas as pd


class Statistics:
    def __init__(self) -> None:
        pass

    def split_mask_step(
        self, row_index, column_index, segmentation_mask, mask, class_level, current_id
    ):
        """Single step in the split mask function that sets the tree/canopy of which the current (row_index, column_index) is part and updates
        mask based on this. Output is whether or not a new tree/canopy was used. It achieves this by checking the left and top pixel of the current pixel.
        If they are part of a tree/canopy, it sets the mask value of this pixel to that value, otherwise it creates a new tree/canopy at that position

        Args:
            row_index (int): row of the current element
            column_index (int): col of the current index
            segmentation_mask (np.matrix(bool)): Segmentation mask produced by the merge_tiled_results function of the Model class
            mask (np.matrix(int)): Mask where each int indicates which tree/canopy that pixelis part of
            class_level (int): 0 for canopy, 1 for tree
            current_id (int): the current id that has to be used when a new instance of tree/canopy is detected at that positions

        Returns:
            bool: whether or not a new instance was created
        """
        if segmentation_mask[row_index, column_index, class_level]:
            top_id = 0
            left_id = 0
            if (
                row_index > 0
                and segmentation_mask[row_index - 1, column_index, class_level]
            ):
                top_id = mask[row_index - 1, column_index]
            if (
                column_index > 0
                and segmentation_mask[row_index, column_index - 1, class_level]
            ):
                left_id = mask[row_index, column_index - 1]

            if top_id > 0 and left_id > 0 and left_id != top_id:
                mask[row_index, column_index] = min(left_id, top_id)
                mask[mask == max(left_id, top_id)] = min(left_id, top_id)
            elif top_id > 0:
                mask[row_index, column_index] = top_id
            elif left_id > 0:
                mask[row_index, column_index] = left_id
            else:
                mask[row_index, column_index] = current_id
                return True

        return False

    def split_mask(self, segmentation_mask):
        """Splits the segmentation mask of the merge_tiled_results of Model into trees and canopies. Both the trees and canopies is a mask matrix,
        where the value at each pixel indicates of which tree/canopy it is a part (0 means no tree/canopy)

        Args:
            segmentation_mask (np.matrix(bool)): Segmentation mask of the merge_tiled_results of Model

        Returns:
            np.matrix(int), np.matrix(int): Mask of the trees (resp. canopies) in the segmentation
        """
        trees = np.zeros(segmentation_mask.shape[:2]).astype(np.int32)
        canopies = np.zeros(segmentation_mask.shape[:2]).astype(np.int32)
        current_tree_id = 1
        current_canopy_id = 1
        for i in range(segmentation_mask.shape[0]):
            for j in range(segmentation_mask.shape[1]):
                increase_id_canopies = self.split_mask_step(
                    i, j, segmentation_mask, canopies, 0, current_canopy_id
                )
                if increase_id_canopies:
                    current_canopy_id += 1

                increase_id_trees = self.split_mask_step(
                    i, j, segmentation_mask, trees, 1, current_tree_id
                )
                if increase_id_trees:
                    current_tree_id += 1

        return trees, canopies

    def split_masks_single(self, image_shape, results, treshold=0.5):
        """Splits the segmentation mask of the predict_file of Model into trees and canopies. Both the trees and canopies is a mask matrix,
        where the value at each pixel indicates of which tree/canopy it is a part (0 means no tree/canopy)

        Args:
            image_shape (np.array(int)): Shape of the input image
            results (Instance): Result as outputted by predict_file
            treshold (float, optional): Treshold to be used for the detection. Defaults to 0.5.

        Returns:
            np.matrix(int), np.matrix(int): Mask of the trees (resp. canopies) in the segmentation
        """
        trees = np.zeros(image_shape[:2]).astype(np.int32)
        canopies = np.zeros(image_shape[:2]).astype(np.int32)
        current_tree_id = 1
        current_canopy_id = 1
        for index in range(len(results.pred_classes)):
            if results.scores[index] < treshold:
                continue
            if results.pred_classes[index] == 1:
                trees[results.pred_masks[index]] = current_tree_id
                current_tree_id += 1
            elif results.pred_classes[index] == 0:
                canopies[results.pred_masks[index]] = current_canopy_id
                current_canopy_id += 1

        return trees, canopies

    def per_instance_stats(self, mask, image, index_instance):
        """Gets some statistics for a specific tree/canopy

        Args:
            mask (np.array(int)): Mask as produced by split_mask_*, either the trees or canopies
            image (np.array(int)): Image
            index_instance (int): The integer for the tree/canopy for which the stats have to be created

        Returns:
            dict: Dictionary with some stats
        """
        boolean_index = mask == index_instance
        positions = np.array(np.where(boolean_index))
        image_values = image[boolean_index]
        instance_stats = {
            "x": np.mean(positions[0, :]),
            "y": np.mean(positions[1, :]),
            "pixel_size": positions.shape[1],
            "red_value": np.mean(image_values[:, 0]),
            "green_value": np.mean(image_values[:, 1]),
            "blue_value": np.mean(image_values[:, 2]),
        }
        return instance_stats

    def run(self, image, result=None, segmentation_mask=None, treshold=0.5):
        """Runs the Statistics module and generates statistics for the given result

        Args:
            image (np.matrix(int)): Image
            result (Instance, optional): Results outputted by the predict_file of Model. Either this or segmentation_mask must be not None. Defaults to None.
            segmentation_mask (np.matrix(bool), optional): Segmentation mask outputted by merge_tiled_results. Either this or result must be not None. Defaults to None.
            treshold (float, optional): Treshold to be used (ignored if segmentation_mask is specified). Defaults to 0.5.

        Returns:
            dict, pd.DataFrame, pd.DataFrame: General stats, per tree stats, per canopy stats
        """
        if segmentation_mask is not None:
            trees, canopies = self.split_mask(segmentation_mask)
        else:
            trees, canopies = self.split_masks_single(image.shape, result, treshold)
            segmentation_mask = np.zeros((image.shape[0], image.shape[1], 2))
            segmentation_mask[:, :, 0] = canopies > 0
            segmentation_mask[:, :, 1] = trees > 0

        general_stats = {
            "n_trees": len(np.unique(trees)),
            "n_canopies": len(np.unique(canopies)),
            "tree_cover": np.count_nonzero(segmentation_mask[:, :, 1])
            / np.prod(image.shape[:2]),
            "canopy_cover": np.count_nonzero(segmentation_mask[:, :, 0])
            / np.prod(image.shape[:2]),
            "tree_canopy_cover": np.count_nonzero(np.any(segmentation_mask, axis=2))
            / np.prod(image.shape[:2]),
        }

        tree_stats = []
        for tree_index in np.unique(trees):
            tree_stats.append(self.per_instance_stats(trees, image, tree_index))

        canopy_stats = []
        for canopy_index in np.unique(canopies):
            canopy_stats.append(self.per_instance_stats(canopies, image, canopy_index))

        return general_stats, pd.DataFrame(tree_stats), pd.DataFrame(canopy_stats)
