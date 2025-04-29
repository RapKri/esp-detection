# ESP-Detection 

ESP-Detection is a lightweight and ESP-optimized project based on [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics), designed for real-time object detection on ESP series chips. It allows you effortlessly train a detection model for specific target and deploy the model on chips easily by [ESP_DL](https://github.com/espressif/esp-dl).

## Overview

ESP-Detection provides a series of ultra-lightweight models along with APIs that enables you to train custom detection models tailored to your specific use cases. The offered models are optimized for efficient deployment on  ESP AI chips, like ESP32-P4 and ESP32-S3. The project also includes example applications such as cat detection, dog detection, and pedestrian detection. 

- Competitive mAP: The single-class model ESPDet-Pico(0.36M parameters, 224 input size) achieves comparable mAP(0.5:0.95) to 80-class model YOLOv11n(2.6M parameters, 640 input size) on cat detection.
- Faster latency: Considering pre-processing and post-processing, espdet_pico achieves >14 FPS on ESP32-P4 and >6 FPS on ESP32-S3 when the input size is 224*224.
- Deploy friendly: A complete deployment solution is provided based on [ESP-DL](https://github.com/espressif/esp-dl), offering users an effortless way to deploy custom models.

## Cat Detection Example

| Model                                                                                                      | Input size | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params<br><sup>(M) | FLOPS<br><sup>(G) | Latency<sup><small>[ESP32-P4](#latency)</small><sup><br><sup>(ms) | Latency<sup><small>[ESP32-S3](#latency)</small><sup><br><sup>(ms) |
|:-----------------------------------------------------------------------------------------------------------|:----------:|:-----------------------:|:------------------:|:------------------:|:-----------------:|:-----------------------------------------------------------------:|:-----------------------------------------------------------------:|
| [espdet_pico_224_224_cat](./examples/cat_detection/espdet_pico_224_224_cat.pt)                             |  224*224   |          69.9           |        88.4        |        0.36        |       0.17        |                               51.4                                |                               126.2                               |
| [espdet_pico_416_416_cat](./examples/cat_detection/espdet_pico_416_416_cat.pt) |  416*416   |          76.6           |        93.4        |        0.36        |       0.60        |                               201.7                               |                               449.5                               |

- **mAP<sup>val</sup>** values are for single-model sing-scale on cat subset of [COCO val2017](https://cocodataset.org/) dataset.

## Updates

- 2025/04/23: esp-detection 1.0.0 is public. [Cat Detection](./examples/cat_detection) is available.


## Installation

```bash
conda create -n espdet python=3.8
conda activate espdet
pip install -r requirements.txt
```

## Quick start

### Step 1: Prepare dataset

Dataset format in esp-detection follows the [YOLO detection dataset format](https://docs.ultralytics.com/datasets/detect/). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

### Step 2: Train and deploy the model

In esp-detection, we provide an all-in-one script ```espdet_run.py``` that streamlines the entire flow. With a single command, users can easily perform model training, export, quantization, and deployment. You can customize the ```espdet_run.sh``` shell script and execute the full pipeline with ```sh espdet_run.sh```.
```bash
python espdet_run.py \
  --class_name mycat \
  --pretrained_path None \
  --dataset "cfg/datasets/coco_cat.yaml" \
  --size 224 \
  --target "esp32p4" \
  --calib_data "deploy/cat_calib" \
  --espdl "espdet_coco_224_224_mycat.espdl" \
  --img "espdet.jpg"
```
- MPS, CPU, Single-GPU and Multi-GPU training are supported in esp-detection. Please refer to [Ultralytics YOLO Docs](https://docs.ultralytics.com/modes/train/) for more information. Specifically, you can set your own train settings according to [Train Settings](https://docs.ultralytics.com/modes/train/#train-settings) from Ultralytics.

### Step 3: Inference on chips
Please refer to [esp-dl/examples](https://github.com/espressif/esp-dl/tree/master/examples) for inference on chips.

```bash
idf.py set-target esp32p4
idf.py flash monitor
```
- If replacing ```espdet.jpg``` with a custom image, ensure its width and height are correctly set in ```app_main.cpp```.

## Feedback

Please submit an [issue](https://github.com/espressif/esp-detection/issues) if you find any problems using our products, and we will reply as soon as possible.

## Reference

- https://github.com/ultralytics/ultralytics