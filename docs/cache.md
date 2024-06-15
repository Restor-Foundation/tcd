# Results caching

The pipeline supports stateful image processing, which is useful for dealing with very large images and processes that may crash. For example if you are trying to predict data over a large city, or a country, it's helpful that (1) if your system crashes you don't need to load it again (2) predictions that are out-of-memory can be stored effectively. This means that you can make predictions over potentially terrabytes of imagery and as long as you have the _hard disk_ space available, you can run that processing on a desktop machine instead of a cloud server.

## Cache overview

The basic functionality of the cache is to:

- Store predictions in geo-referenced formats
- Track which areas of the input image have already been processed
- Act as a usable model output for other things

## Stateful processing

By "stateful" we mean that if you run the pipeline twice on the same image, predictions won't be re-run.

We do this very simply: by tracking the number of tiles from the input that have been processed. This assumes that you use the same settings to re-run prediction, otherwise we would have to check every area of the input image and check if it had been predicted. This is difficult to do reliably with geospatial bounds, because of coordinate precision and while we can do it with pixels for smaller images, on larger images it becomes more complicated (e.g. what is the pixel location for a tile at the scale of an entire country?)

So, provided you've used the same settings from the same version of the software, this approach works well. When prediction starts, we load the information from the cache folder, stored in the `tiles.jsonl` file:

```json
{"image": "/home/josh/code/tcd/data/5c15321f63d9810007f8b06f_10_00000.tif", "classes": ["canopy", "tree"], "suffix": "_instance"}
{"tile_id": 1, "bbox": [0.0, 0.0, 1024.0, 1024.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
{"tile_id": 2, "bbox": [512.0, 0.0, 1536.0, 1024.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
{"tile_id": 3, "bbox": [1024.0, 0.0, 2048.0, 1024.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
{"tile_id": 4, "bbox": [0.0, 512.0, 1024.0, 1536.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
{"tile_id": 5, "bbox": [512.0, 512.0, 1536.0, 1536.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
{"tile_id": 6, "bbox": [1024.0, 512.0, 2048.0, 1536.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
{"tile_id": 7, "bbox": [0.0, 1024.0, 1024.0, 2048.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
{"tile_id": 8, "bbox": [512.0, 1024.0, 1536.0, 2048.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
{"tile_id": 9, "bbox": [1024.0, 1024.0, 2048.0, 2048.0], "cache_file": "/home/josh/code/tcd/temp/5c15321f63d9810007f8b06f_10_00000_cache/instances_instance.shp"}
```

`JSONL` is a plaintext format where each line of the file is a valid JSON document. The advantage is that we can append to it efficiently, without needing to load and store the whole file.

The first line in the file contains metadata about the prediction:

- The source image
- What classes the model uses
- Cache file suffixes

The remaining lines contain:

- The tile index
- The bounding box from the source image
- What file the results were saved to

The last point is important. For instance segmentation, all the results can normally be saved to a single ShapeFile. For semantic segmentation, we store results into a tiled GeoTIFF and if a prediction falls across a tile boundary it is split between tiles.

## Cache "size"

If you call the `len()` property on a cache, it will return the number of tiles that have been processed, determined by the number of lines in the metadata file (minus one).

## Storing data

Cache objects have a uniform interface. They accept (1) a prediction and (2) a source location in the image, plus in principle any other metadata about that prediction. The subclass of `ResultsCache` is responsible for serialising the information in an appropriate way.

## Loading data

The two default caches (`ShapefileInstanceCache` and `GeotiffSemanticCache`) are designed to produce usable data without any further processing. You can load the prediction GeoTIFF straight into GIS software, and same goes for the shapefile. They are stored with the same coordinate reference information as the image used for predictions.

If you're predicting in code, the pipeline will output a `Results` object which is meant to be a simple interface for visualisation and basic analysis. So, the cache has to return results that can be converted into `Results`.

## Handling geospatial data

### Semantic segmentation

For semantic segmentation, we store results drectly as a GeoTIFF image which has the advantage that (a) it has geospatial information built in, and (b) we can easily index into it with pixel coordinates that are aligned with the source image. For example, if we predict a tile with bounds (0,0,1024,1024) in the source image we can write _in pixel coords_ the same data into the cached results.

### Instance segmentation

For instance segmentation things get a bit more complicated. We cache polygons to a Shapefile, which is great because it's a portable format that virtually all GIS software supports. This means that we have to transform the predictions into geospatial coordinates before saving to the file.

If you've ever used `rasterio` then you'll have probably seen `transform`s. For example:

```python
import rasterio

with rasterio.open("../data/5c15321f63d9810007f8b06f_10_00000.tif") as src:
    print(src.transform)
    print(type(src.transform))

# Returns
| 0.10, 0.00,-46189.54|
| 0.00,-0.10, 4719172.21|
| 0.00, 0.00, 1.00|
<class 'affine.Affine'>
```

A `transform`, as the name suggests, transforms points between different coordinate systems. If you've ever done any linear algebra - and in particular have used [transformation matrices]() before, you can spot some features: `0.1` is the resolution of the image in "units" per pixel, typically metres or degrees; this is a _scale factor_. Since we're working in 2D, the third row makes this an "augmented" matrix, which lets us tack on a translation vector in the same matrix (the third colummn). Those two large numbers are the origin of the image and so this matrix tells us, given a point `x = (x', y', 1)` in pixels, where it should map to in _world_ coordinates. In this case:

```latex
X_world = 0.1*x' - 46189.54
Y_world = -0.1*y' + 4719172.2
```

we then rely on the definition of the particular coordinate reference system (CRS) of the image to interpret where that location is in the world.

Our model knows _nothing_ about this. It takes an array of numbers and it outputs predictions in pixel coordinates. We have to convert these local tile coordinates into coordinates of the source image (by addding the origin of the tile) and then convert these "global" image coordinates into world coordinates. And to do _that_ we apply the above affine transform to every coordinate in the polygon. Fortunately `shapely` has a function to do this (`shapely.affinity.affine_transform`):

```python
from shapely.affinity import affine_transform

polygon # in pixels

t = image.transform
transform = [t.a, t.b, t.d, t.e, t.xoff, t.yoff]
world_polygon = affine_transform(polygon, transform)
```

All that to say that `image.transform` allows us to map from pixels into world coordinates. What about the other way?

You may have used `Windows` in `rasterio`, which is a way of specifying a rectangular region that we want to read from or write into. If we have a geospatial region and we want to select the part of the image that we're interested in, we can use the `rasterio.windows.from_bounds` method which takes the bounding box of the region and the image transform. Another really useful method is `geometry_window` which takes a shape, in CRS units, and returns a window _in pixels_ that bounds the shape.

The way we do this is by _inverting_ the transformation matrix (and unsurprisingly `rasterio` does this by specifiying a forward or backward direction when translating). Conveniently, the `affine.Affine` class supports inversion with the `~` operator (`__inv__`), so we can map polygons to pixel coordinates like this:

```python

world_polygon # in CRS units

t = ~image.transform
transform = [t.a, t.b, t.d, t.e, t.xoff, t.yoff]
polygon = = affine_transform(world_polygon, transform)
```

Inevitably this involves some loss of precision converting between coordinates, but it should not be significant unless you repeatedly transform back and forth. Primarily we convert to image coordinates to generate masks from the shapefile (rasterisation).
