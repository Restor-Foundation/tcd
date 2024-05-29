# Predicting tree cover in images

## Image formats

Preferrably, your images should be in [GeoTIFF format](https://www.ogc.org/standard/geotiff/). GeoTIFFs are a special type of TIFF image that contain geoferefencing information - the coordinates of the image and the spatial scale for each pixel. We recommend that images you provide to the model are at a scale of 10 cm/px; we provide tools to do this for you, but we need to know the scale of the input image in order for it to work well.

Most photogrammetry tools like Open Drone Map, DJI Terra, Pix4D, etc will export orthomosaics in GeoTIFF format.

Another advantage of providing geo-referenced imagery is that we can use other inputs like shapefiles to limit the area of analysis - for example, if you want to measure canopy cover over a plot of land, we need to know the extent of that plot. Restor's pipeline is capable of exporting predictions in georeferenced formats too, so you can easily import them into GIS software for analysis. This includes segmentation maps and polygon (shapefile) output for instance segmentation models.

While our pipeline can run on non-georeferenced images (the models themselves have no explicit knowledge of scale), results may vary. Our models were trained on images of a fixed size with the idea that they should "learn" what scale trees tend to be. You'll probably find that the models work OK, but try and get the scale around 10cm/px.

## Setup and running your first prediction

The `predict` script is the simplest way to run predictions on a single image. You need to specify what sort of prediction results you need (e.g. `instance` vs `semantic` segmentation).

```bash
python predict.py instance \
  data/5c15321f63d9810007f8b06f_10_00000.tif \
  test_predictions_instance
```

```bash
python predict.py semantic \
  data/5c15321f63d9810007f8b06f_10_00000.tif \
  test_predictions_semantic
```

The default settings assume that you have a moderately powerful computer with a GPU. You can absolutely run predictions on a CPU and PyTorch also supports the `mps` backend for Mac OS X, it'll just take a little longer.

You can provide more options to the script if you need to, here is the help output:

```bash
usage: predict.py [-h] {instance,semantic} input output ...

positional arguments:
  {instance,semantic}
  input                Path to input image (i.e. GeoTIFF)
  output               Path to output folder for predictions, will be created if it doesn't exist
  options              Configuration options to pass to the pipeline, formatted as <key>=<value> with spaces between options.

options:
  -h, --help           show this help message and exit

```

For example, if you wanted to change the tile size to the model, you could run:

```bash
python predict.py \
  instance data/5c15321f63d9810007f8b06f_10_00000.tif \ 
  test_predictions_instance \
  data.tile_size=1024
```

Any options provided after the positional arguments (i.e. the task type, input and output) are parsed as space-separated key/value pairs. If you need to provide an option that has spaces, you should escape it with quotes. For more information about configuration, see the [config](config.md) page.

If this is the first time you've run prediction, the weights for`restor/tcd-segformer-mit-b0` will be downloaded, which is a variant of Segformer, a lightweight semantic segmentation [architecture](https://huggingface.co/docs/transformers/model_doc/segformer) architecture that provides good performance with relatively few model parameters (we trained models up to `mit-b5` but found diminishing returns beyond `mit-b3`). We provide this tile as a test image, but you're free to swap in your own data. The pipeline will automatically deal with large images by splitting into tiles, processing them separately and then combining the results.

Depending on the mode, the script will produce a number of outputs in the selected folder, including:

- Segmentation maps
- Instance segmentation polygons, exported as a shapefile
- Other reporting information

## Spatial information in images

A common cause for confusion in ML for Earth observation is understanding how model performance varies with tile size and image spatial resolution or ground sample distance.

There are a several factors which are somewhat unique to remote sensing data which trip users and researchers up all the time.

### Image resolution

Remotely sensed images from aerial or satellite platforms have an intrinsic resolution in the units are metres or centimetres per pixel. This can be calculated as follows:

- The field of view of a camera is determined by the focal length of its lens, and the physical dimensions of the sensor. You will often see this quoted as HFoV or VFoV (horizontal or vertical field of view) in degrees.
- The range to the target determines, via some simple trigonometry, what the field of view is in metric units (i.e. distance)
- If you know the extent of the frame as a distance, the number of pixels on the sensor determines the ground sample distance in metres per pixel.

If you are using a drone to capture imagery, your flight planning software probably has some utility to figure out how high you need to fly to produce an orthomosaic with a minimum resolution. The higher the resolution, the lower you need to fly and the longer it will take to cover a survey area (in terms of battery consumption and number of flights).

Here are some general image resolutions that you can expect to see:

- Drones - capable of sub-centimetre imagery but typically produce orthomosaics on the order of a few cm/px
- Aerial surveys - planes are expensive to charter, so these data are normally obtained through national/regional mapping efforts. Often combines LIDAR scanning. Optical resolution can be very high, but it's commong to see 0.1-0.5 m/px in public data. LIDAR data is generally provided at lower resolution, 0.5-2 m/px.
- Satellite images - availability varies from open access (Sentinel 2) to paid (e.g. Airbus Neo). Generally the higher the resolution, the more you pay. The more frequent the imagery, the more you pay. Image resolution varies, but is limited by publicly available satellite technology. The best commercially available data is around 0.3 m/px while 0.5-2 m/px is more common.
- Sentinel 2 - it's worth discussing S2 separately because it's a free data source that covers the whole globe at regular (up to 12 day) intervals. As it's free and the data are high quality, it is frequently used in research and there is a huge body of literature on tree detection/assessment alone. However, at 10 m/px, it simply isn't high resolution enough to see individual tree crowns for all but the largest specimens.

#### Aerial and satellite imagery comparison

The figure below shows the Beckenhof area of Zurich, just North of Hauptbahnhof which is visible in the lower left. The images were captured by three aerial imaging platforms at approximately the same time. On the left is Sentinel 2 at 10 m, in the middle Planet (Dove/PlanetScope) at 3 m, Maxar (ESRI Wayback) at 0.3-0.5 m and SwissTopo at 0.1 m.

[ Images here ]

In the low resolution Sentinel image, it is possible to make out the difference between the park at Beckenhof and buildings. When we increase to 3m, we start to be able to make out road structures, shadowing is visible and we start to be able to see canopy structure. By the time we get to Maxar's Worldview, we can easily see individual branches and the quality is similar to what you'd see in a populated area on Google Maps. The highest resolution image we have here is from Swisstopo at 0.1 m and even people are visible.

Hopefully it's clear from these images that having high resolution data greatly impacts how well you can predict tree cover. Our models were trained almost exclusively on labelled and re-sampled imagery captured by drones. Drones are increasingly affordable and are a practical solution for surveying small sites.

### How models interpret images

Now we've seen some examples of images at different resolutions, what's best? First, we need to understand how models perform predictions and what sort of pre-processing takes place.

The ML models we use have no notion of scale, they simply see a 3D array of pixels (height, width, RGB channels). During training, we can provide some consistency by only allowing the model to see images around a fixed resolution - in our case, 10 cm/px. This means we need to be careful about any re-scaling or re-sizing that happens at prediction or training time.

There are a few effects that can trip you up:

- Some models have a pipeline step that will re-size an image if it's too big. For example, there is a configuration parameter in the Detectron2 library that handles this. In a lot of situations this is sensible behaviour. Object detection models for terrestrial scenes do not need very high resolution images and modern cameras have a lot of megapixels. For remote sensing imagery this is a problem, because it means that the _spatial_ resolution that the model sees is a function of the size of the input image.
- Some models have fixed output sizes. Both Mask-RCNN and Segformer predict output masks at a fixed size (in pixels) which is re-scaled to match the size of the source image. This is for performance reasons as it's not always necessary to predict masks at native resolution. However if you want to detect small objects, choosing a large input image size means that your predictions (in absolute pixels) might be very small before they're scaled up.
- Object detection models usually have a fixed number of predictions. Most of these models were trained on terrestrial images with relatively few objects in the image (< 100). But a single image of an orchard or plantation from the air might easily contain several hundred trees.
- Images that are too small, in the sense of spatial coverage (a 256 px image at 0.1 m/px is only 25 m wide), might not contain enough context for the models to predict well.

We try to get around these problems as follows:

- We disable all pre-processing scaling before the image is passed to the model. The tiling routine in the pipeline makes sure that tiles for prediction are at the desired size (in pixels) and at the desired spatial resolutino (in m/px). The only transformation we make is a normalisation step.
- For Mask-RCNN, we set the maximum object count to be quite high (256) and we warn you if the model has predicted the maximum number of high-confidence objects.
- Despite the mask output for Mask-RCNN being very small (by default, just 28x28 pixels per object), this turns out to be reasonable for tree detection.
- We recommend that you don't predict on tiles that are too large or too small. 1024 px with an overlap of 256 px is fine for most use-cases.

## Predicting using the library

Once you've installed the pipeline, a simple way of testing it is to launch a Jupyter notebook and run:

```python
from tcd_pipeline.pipeline import Pipeline

# Assuming you run from the root directory
pipeline = Pipeline("semantic")

image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
res = runner.predict(image_path)
res # You should see a plot below the cell
```

## Prediction caching

Our pipeline handles huge (out of memory) images by using [windowed reading and writing](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html). Since you don't need to store the entire image (or its predictions) in memory, you can process arbitrarily large images provided you have enough hard disk space to store the predictions.

Consider the city of Zurich as an example. Swisstopo provides 10 cm tiles covering the administrative boundary of the city, producing an image that's around 20 Gigapixels when viewed as a virtual raster. The compressed RGB images are around 7GB. If we wanted to hold the raw predictions for this data in memory, we would need around 20 GB of RAM for a binary classification using 8-bit confidence values. That's possible on a modern machine, but when you then consider predicting with a multi-class model, over the wider Canton, or the whole country, you would need a very expensive server with a huge amount of RAM.

Windowed reading is used during tiling so that only the part of the image that is currently being predicted is loaded into memory and each prediction is stored to disk afterwards. For instance segmentation this is quite simple because we can store the detected objects in a compact way (as serialised polygons).

For semantic segmentation, we store predictions directly to a tiled GeoTIFF, referenced to the source image. The cached tiles are then copied to the prediction output folder and a virtual raster (VRT) is automatically created. The output prediction tiles are chosen to be large out of convenience, so your output folder doesn't have thousands of tiny predictions. During inference, tiles are stored uncompressed to save time when writing. Once prediction is complete, the tiles are compressed before being copied.

You can customise the cache folder during prediction using the `postprocess.cache_folder=<path to cache>` option (e.g. if you have a scratch drive). By default, the cache will be cleared when prediction is complete.

During predictions, you need approximately one byte per pixel of storage space available for binary models.

## Analysing huge images

With smaller images - say a single drone flight - you can take advantage of processing in the pipeline for visualising results. However, for very large images (e.g. city scale or above) you are likely to run into RAM limitations. We recommend that you process very large segmentation maps in dedicated GIS software.

## Pipeline outputs

There are a few concepts that it's helpful to understand when using the pipeline.

- We introduced the `Pipeline` class above. This loads models and provides user-friendly interfaces for predicting directly from an image path. Most of the time you'll interact with the pipeline using this object.
- All models sub-class `Model`, an abstract class that provides support for tiled inference and handles post-processing of results. You rarely need to interact with the "raw" model itself, instead you call `pipeline.predict`.
- Predictions are passed to a `PostProcessor` which caches results to disk and then reconstructs the final result once inference is complete. This step also allows you to recover the state of a prediction job, for example if you have a huge image and something fails part-way through.
- The pipeline returns a `Result` object which offers methods for visualisation and storing of results. You can visualise and save masks, clip results with a geometry file (such as a GeoJSON or Shapefile) and compute basic statistics on the data. From a result object you can also serialise data in standard formats such as exporting instances to a Shapefile, or a canopy mask to a GeoTIFF.

Like the basic example above, the details are hidden for simplicity you can simply call `predict` and then work with the `Result` that's returned. Most of the methods that you can call on a model output are common to both Instance and Semantic segmentation, like `visualise`.

## Working with results

### Storing and loading

### Visualising

### Serialisation

### Mask generation

By default, the semantic segmentation pipeline will output a GeoTIFF with prediction confidence per pixel. The confidence map is an 8-bit image with predictions normalised between 0-255; the `nodata` value in the image is `0` so if you load the image into software like QGIS, tree-free areas should be rendered as transparent. You can threshold this image to produce a binary canopy classification.

For small or medium size images, you can run the `save_masks()` method on the returned result. For instance segmentation models this will rasterise the predicted polygons.

## Generating reports

We provide a convenience script that will process imagery and generate a PDF report (alongside other data products for analysis). This is found in the `tools` folder.

You can check the available options by running `python tools/predict_report.py -h`:

```bash
usage: predict_report.py [-h] -i IMAGE [-g GEOMETRY] [-s TILE_SIZE] [-o OUTPUT] [-r] [--semantic_only] [--overwrite] [--gsd GSD] [--instance_seg INSTANCE_SEG] [--semantic_seg SEMANTIC_SEG]

options:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Input image (GeoTIFF orthomosasic)
  -g GEOMETRY, --geometry GEOMETRY
                        Input shapefile (GeoJSON or Shapefile)
  -s TILE_SIZE, --tile-size TILE_SIZE
                        Tile size
  -o OUTPUT, --output OUTPUT
                        Working and output directory
  -r, --resample        Resample image
  --semantic_only       Only perform semantic segmentation
  --overwrite           Overwrite existing results, otherwise use old ones.
  --gsd GSD
  --instance_seg INSTANCE_SEG
                        Instance segmentation config
  --semantic_seg SEMANTIC_SEG
                        Semantic segmentation config

```

Many of these options have defaults - for example we normally predict using a GSD of 0.1 m/px and the model configurations are also set for you. As with the model pipeline, you can adjust the tile size manually here.

You should provide an image (`-i`) which is usually an orthomosaic. If you have a shapefile or GeoJSON file that defines your plot area, you can provide it with `-g` and predictions will be limited/cropped to this area. The prediction tool will analyse your geometry file and produce a separate report for each geometry in the file, so for example if you have a large orthomosaic and multiple plots, prediction will happen once and the results will be cropped individually for each region. You should provide an output path (`-o`) where the results and intermediate files will be stored.
