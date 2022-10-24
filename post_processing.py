import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.affinity import translate


class Bbox:
    """A bounding box with integer coordinates."""

    def __init__(self, minx, miny, maxx, maxy):
        """Initializes the Bounding box

        Args:
            minx (float): minimum x coordinate of the box
            miny (float): minimum y coordinate of the box
            maxx (float): maximum x coordiante of the box
            maxy (float): maximum y coordinate of the box
        """
        self.minx = (int)(minx)
        self.miny = (int)(miny)
        self.maxx = (int)(maxx)
        self.maxy = (int)(maxy)
        self.bbox = (self.minx, self.miny, self.maxx, self.maxy)
        self.width = self.maxx - self.minx
        self.height = self.maxy - self.miny

    def overlap(self, other):
        """Checks whether this bbox overlaps with another one

        Args:
            other (Bbox): other bbox

        Returns:
            bool: Whether or not the bboxes overlap
        """
        if (
            self.minx >= other.maxx
            or self.maxx <= other.minx
            or self.maxy <= other.miny
            or self.miny >= other.maxy
        ):
            return False
        return True

    def __str__(self):
        return f"Bbox(minx={self.minx:.4f}, miny={self.miny:.4f}, maxx={self.maxx:.4f}, maxy={self.maxy:.4f})"


class ProcessedInstance:
    """Contains a processed instance that is detected by the model. Contains the score the algorithm gave, a polygon for the object,
    a bounding box and a local mask (a boolean mask of the size of the bounding box)
    """

    def __init__(self, score, polygon, bbox, class_index):
        """Initializes the instance

        Args:
            score (float): score given to the instance
            polygon (MultiPolygon): a shapely MultiPolygon describing the segmented object
            bbox (Bbox): the bounding box of the object
            class_index (int): the class index of the object
        """
        self.score = score
        self.polygon = polygon
        self.bbox = bbox
        self.class_index = class_index

        new_poly = translate(self.polygon, xoff=-self.bbox.minx, yoff=-self.bbox.miny)
        shape_local_mask = (self.bbox.height, self.bbox.width)
        self.local_mask = rasterio.features.rasterize(
            [new_poly], out_shape=shape_local_mask
        ).astype(bool)

    def get_pixels(self, image):
        """Gets the pixel values of the image at the location of the object

        Args:
            image (np.array(int)): image

        Returns:
            np.array(int): pixel values at the location of the object
        """
        return image[self.bbox.miny : self.bbox.maxy, self.bbox.minx : self.bbox.maxx][
            self.local_mask
        ]

    def __str__(self):
        return f"ProcessedInstance(score={self.score:.4f}, class={self.class_index}, {str(self.bbox)})"


class ProcessedResult:
    """A processed result of a model. It contains all trees separately and also a global tree mask, canopy mask and image"""

    def __init__(self, image, trees=None, tree_mask=None, canopy_mask=None):
        """Initializes the Processed Result

        Args:
            image (np.array(int)): the image
            trees (List[ProcessedInstance], optional): List of all trees. Defaults to None.
            tree_mask (np.array(bool), optional): Boolean mask for the trees. Defaults to None.
            canopy_mask (np.array(bool), optional): Boolean mask for the canopy. Defaults to None.
        """
        self.trees = trees
        self.tree_mask = tree_mask
        self.canopy_mask = canopy_mask
        self.image = image

    def visualise(
        self, color_trees=(0.8, 0, 0), color_canopy=(0, 0, 0.8), alpha=0.4, **kwargs
    ):
        """Visualizes the result

        Args:
            color_trees (tuple, optional): rgb value of the trees. Defaults to (0.8, 0, 0).
            color_canopy (tuple, optional): rgb value of the canopy. Defaults to (0, 0, 0.8).
            alpha (float, optional): alpha value. Defaults to 0.4.
        """
        fig, ax = plt.subplots(**kwargs)
        plt.axis("off")
        ax.imshow(self.image)

        canopy_mask_image = np.zeros(
            (self.image.shape[0], self.image.shape[1], 4), dtype=float
        )
        canopy_mask_image[self.canopy_mask == 1] = list(color_canopy) + [alpha]
        ax.imshow(canopy_mask_image)

        tree_mask_image = np.zeros(
            (self.image.shape[0], self.image.shape[1], 4), dtype=float
        )
        tree_mask_image[self.tree_mask == 1] = list(color_trees) + [alpha]
        ax.imshow(tree_mask_image)

        plt.show()

    def __str__(self) -> str:
        canopy_cover = np.count_nonzero(self.canopy_mask) / np.prod(
            self.image.shape[:2]
        )
        tree_cover = np.count_nonzero(self.tree_mask) / np.prod(self.image.shape[:2])
        return f"ProcessedResult(n_trees={len(self.trees)}, canopy_cover={canopy_cover:.4f}, tree_cover={tree_cover:.4f})"


class PostProcessor:
    """Processes the result of the modelRunner"""

    def __init__(self, config):
        """Initializes the PostProcessor

        Args:
            config (DotMap): the configuration
        """
        self.config = config

    def _mask_to_polygon(self, mask):
        """Converts the mask of an object to a MultiPolygon

        Args:
            mask (np.array(bool)): Boolean mask of the segmented object

        Returns:
            MultiPolygon: Shapely MultiPolygon describing the object
        """
        all_polygons = []
        for shape, _ in features.shapes(mask.astype(np.int16), mask=mask):
            all_polygons.append(shapely.geometry.shape(shape))

        all_polygons = shapely.geometry.MultiPolygon(all_polygons)
        if not all_polygons.is_valid:
            all_polygons = all_polygons.buffer(0)
            # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            # need to keep it a Multi throughout
            if all_polygons.type == "Polygon":
                all_polygons = shapely.geometry.MultiPolygon([all_polygons])
        return all_polygons

    def _get_proper_bbox(self, image, bbox=None):
        """Gets the proper bbox of an image given a Detectron Bbox

        Args:
            image (np.array(int)): the image
            bbox (Detectron.BoundingBox): Original bounding box of the detectron algorithm. Defaults to None (bbox is entire image)

        Returns:
            Bbox: Bbox with correct orientation compared to the image
        """
        if bbox is None:
            minx, miny = 0, 0
            maxx, maxy = image.shape[0], image.shape[1]
        else:
            miny, minx = image.index(bbox.minx, bbox.miny)
            maxy, maxx = image.index(bbox.maxx, bbox.maxy)
        # Sort coordinates if necessary
        if miny > maxy:
            miny, maxy = maxy, miny

        if minx > maxx:
            minx, maxx = maxx, minx

        return Bbox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def non_max_suppression(self):
        pass

    def remove_edge_predictions(self):
        pass

    def process_untiled_result(self, result, image, treshold=0.5):
        """Processes results outputted by Detectron without tiles

        Args:
            results (Instances): Results predicted by the detectron model
            image (DatasetReader): Image
            treshold (float, optional): Treshold for adding the detected objects. Defaults to 0.5.

        Returns:
            ProcessedResult: ProcessedResult of the segmentation task
        """
        return self.process_tiled_result([[result, None]], image, treshold)

    def _collect_tiled_result(self, results, image, treshold=0.5):
        """Collects all segmented objects that are predicted and puts them in a ProcessedResult. Also creates global masks for trees and canopies

        Args:
            results (List[[Instances, Detectron.BoundingBox]]): Results predicted by the detectron model
            image (DatasetReader): Image
            treshold (float, optional): Treshold for adding the detected objects. Defaults to 0.5.

        Returns:
            List[[ProcessedInstance, int]], np.array(bool), np.array(bool): Returns a list containing all ProcessedResults together with the tile in
                                                                            which they were discovered,
                                                                    global mask for the canopy, global mask for the tree
        """
        untiled_instances = []

        canopy_mask = np.zeros((image.height, image.width), dtype=bool)

        tree_mask = np.zeros((image.height, image.width), dtype=bool)

        for tile, result in enumerate(results):
            instances, bbox = result
            proper_bbx = self._get_proper_bbox(image, bbox)

            for instance_index in range(len(instances)):
                if (
                    instances.scores[instance_index] < treshold
                ):  # remove objects below treshold
                    continue

                mask = instances.pred_masks[instance_index].cpu().numpy()
                class_idx = int(instances.pred_classes[instance_index])
                mask_height, mask_width = mask.shape
                polygon = self._mask_to_polygon(mask)
                polygon = translate(
                    polygon, xoff=proper_bbx.minx, yoff=proper_bbx.miny
                )  # translate the polygon to match the image

                if self.config.data.classes[class_idx] == "canopy":
                    canopy_mask[
                        proper_bbx.miny : proper_bbx.miny + mask_height,
                        proper_bbx.minx : proper_bbx.minx + mask_width,
                    ][mask] = True
                else:
                    bbox_instance_tiled = (
                        instances.pred_boxes[instance_index].tensor[0].cpu().numpy()
                    )
                    bbox_instance = Bbox(
                        minx=proper_bbx.minx + bbox_instance_tiled[0],
                        miny=proper_bbx.miny + bbox_instance_tiled[1],
                        maxx=proper_bbx.minx + bbox_instance_tiled[2],
                        maxy=proper_bbx.miny + bbox_instance_tiled[3],
                    )

                    new_instance = ProcessedInstance(
                        class_index=class_idx,
                        polygon=polygon,
                        bbox=bbox_instance,
                        score=instances.scores[instance_index],
                    )

                    tree_mask[
                        proper_bbx.miny : proper_bbx.miny + mask_height,
                        proper_bbx.minx : proper_bbx.minx + mask_width,
                    ][mask] = True

                    untiled_instances.append([new_instance, tile])

        return untiled_instances, canopy_mask, tree_mask

    def process_tiled_result(self, results, image, treshold=0.5):
        """Processes the result of the detectron model when the tiled version was used

        Args:
            results (List[[Instances, Detectron.BoundingBox]]): Results predicted by the detectron model
            image (np.array(int)): Image
            treshold (float, optional): Treshold for adding the detected objects. Defaults to 0.5.

        Returns:
            ProcessedResult: ProcessedResult of the segmentation task
        """
        untiled_instances, canopy_mask, tree_mask = self._collect_tiled_result(
            results, image, treshold
        )

        merged_instances = []

        for instance_index, instance_info in enumerate(untiled_instances):
            instance, tile = instance_info
            other_instance_index = instance_index + 1
            while other_instance_index < len(untiled_instances):

                other_instance, other_tile = untiled_instances[other_instance_index]
                # instance bbox overlap is not strictly necessary, just avoids the more expensive intersects case when it is not necessary
                # TODO: if tile == other_tile, the trees should maybe be changed by canopy? Probably first implement the NMS
                # TODO: do we require a minimum overlap before the objects are combined? Probably first implement the NMS
                if (
                    instance.bbox.overlap(other_instance.bbox)
                    and tile != other_tile
                    and instance.class_index == instance.class_index
                    and instance.polygon.intersects(other_instance.polygon)
                ):  # if the bbox overlap -> make it one object

                    bbox1 = instance.bbox
                    bbox2 = other_instance.bbox
                    size1 = instance.polygon.area
                    size2 = other_instance.polygon.area

                    instance = ProcessedInstance(
                        class_index=instance.class_index,
                        polygon=instance.polygon.union(other_instance.polygon),
                        bbox=Bbox(
                            minx=min(bbox1.minx, bbox2.minx),
                            miny=min(bbox1.miny, bbox2.miny),
                            maxx=max(bbox1.maxx, bbox2.maxx),
                            maxy=max(bbox1.maxy, bbox2.maxy),
                        ),
                        score=(size1 * instance.score + size2 * other_instance.score)
                        / (size1 + size2),
                    )
                    del untiled_instances[other_instance_index]
                    # go back to the first instance: could be that by combining the objects, previous objects can now also be added
                    other_instance_index = instance_index + 1
                else:
                    other_instance_index = other_instance_index + 1

            merged_instances.append(instance)

        return ProcessedResult(
            image=image.read().transpose(1, 2, 0),
            trees=merged_instances,
            tree_mask=tree_mask,
            canopy_mask=canopy_mask,
        )
