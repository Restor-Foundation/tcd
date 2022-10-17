import numpy as np
import pandas as pd


class Statistics:
    def __init__(self) -> None:
        pass

    def split_mask_step(
        self, row_index, column_index, segmentation_mask, mask, class_level, current_id
    ):
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

    def split_masks_single(self, image, results, treshold=0.5):
        trees = np.zeros(image.shape[:2]).astype(np.int32)
        canopies = np.zeros(image.shape[:2]).astype(np.int32)
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
        positions = np.array(np.where(mask == index_instance))
        image_values = image[mask == index_instance]
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
        if segmentation_mask is not None:
            trees, canopies = self.split_mask(segmentation_mask)
        else:
            trees, canopies = self.split_masks_single(image, result, treshold)
            segmentation_mask = np.zeros((image.shape[0], image.shape[1], 2))
            segmentation_mask[:, :, 0] = canopies > 0
            segmentation_mask[:, :, 1] = trees > 0

        general_stats = {
            "n_trees": np.max(trees),
            "n_canopies": np.max(canopies),
            "tree_cover": np.count_nonzero(segmentation_mask[:, :, 1])
            / np.prod(image.shape[:2]),
            "canopy_cover": np.count_nonzero(segmentation_mask[:, :, 0])
            / np.prod(image.shape[:2]),
            "tree_canopy_cover": np.count_nonzero(np.any(segmentation_mask, axis=2))
            / np.prod(image.shape[:2]),
        }

        tree_stats = []
        for tree_index in range(1, general_stats["n_trees"] + 1):
            tree_stats.append(self.per_instance_stats(trees, image, tree_index))

        canopy_stats = []
        for canopy_index in range(1, general_stats["n_canopies"] + 1):
            canopy_stats.append(self.per_instance_stats(canopies, image, canopy_index))

        return general_stats, pd.DataFrame(tree_stats), pd.DataFrame(canopy_stats)
