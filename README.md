# Faster-RCNN-COCO_TF

This repo is a modified fork of [Faster-RCNN_TF by smallcorgi](https://github.com/smallcorgi/Faster-RCNN_TF) which implements [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

The repo has been modified for training on [MS COCO](http://cocodataset.org/#home), in particular the 2014 dataset, as well as visualizing on a headless server. A pretrained model is also included below. 

### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Installation

1. Clone the Faster R-CNN repository
	```bash
	# Make sure to clone with --recursive
	git clone --recursive https://github.com/dxyang/Faster-RCNN-COCO_TF.git
	```

2. Build pycocotools modules
	```bash
	cd $FRCN_ROOT/lib
	git clone https://github.com/cocodataset/cocoapi.git
	cd cocoapi/PythonAPI
	make
	cd ../..
	mv cocoapi/PythonAPI/pycocotools pycocotools
	rm -rf cocoapi
	```

3. Build the Cython modules
	```bash
	cd $FRCN_ROOT/lib
	make
	```


### Training Model
1. Install gsutil if you haven't already
	```bash
	curl https://sdk.cloud.google.com | bash
	```

2. Download the training, validation, test data for MS COCO
	```bash
	cd $FRCN_ROOT/data
	mkdir coco; cd coco
	mkdir images; cd images
	mkdir train2014
	mkdir test2014
	mkdir val2014
	gsutil -m rsync gs://images.cocodataset.org/train2014 train2014
	gsutil -m rsync gs://images.cocodataset.org/test2014 test2014
	gsutil -m rsync gs://images.cocodataset.org/val2014 val2014
	```

3. Download the annotations for MS COCO and unzip
	```bash
	cd $FRCN_ROOT/data
	gsutil -m rsync gs://images.cocodataset.org/annotations coco
	cd coco
	unzip annotations_trainval2014.zip
	unzip image_info_test2014.zip
	rm *.zip
	```

4. Download the annotations for the 5000 image minival subset of COCO val2014 as mentioned [here](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data)
	```bash
	cd $FRCN_ROOT/data/coco/annotations
	wget https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip
	wget https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip
	unzip instances_minival2014.json.zip; rm instances_minival2014.json.zip
	unzip instances_valminusminival2014.json.zip; rm instances_valminusminival2014.json.zip
	```

5. Download the pre-trained ImageNet model [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)
	```bash
	cd $FRCN_ROOT
	wget https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy
	mkdir data/pretrain_model
	mv VGG_imagenet.npy data/pretrain_model/VGG_imagenet.npy
	```

6. Create an output directory for log files
	```bash
	cd $FRCN_ROOT
	mkdir experiments/logs
	```

7. Run script to train and test model
	```bash
	cd $FRCN_ROOT
	./experiments/scripts/faster_rcnn_end2end.sh $DEVICE $DEVICE_ID VGG16 coco
	```
  - DEVICE is either cpu/gpu

### Testing Model
Run the following command.

```bash
python ./tools/test_net.py \
		--device gpu \
		--device_id 0 \
		--weights output/faster_rcnn_end2end/coco_2014_train/VGGnet_fast_rcnn_iter_490000.ckpt \
		--cfg experiments/cfgs/faster_rcnn_end2end.yml \
		--imdb coco_2014_minival \
		--network VGGnet_test	\
		--vis False
```

- Changing ```vis``` to ```True``` will save images with all detections above 0.8 for every image in the testing set.
- The checkpoint files folder contains the following:
	```bash
	cd output/faster_rcnn_end2end/coco_2014_train
	ls
	# VGGnet_fast_rcnn_iter_490000.ckpt.data-00000-of-00001
	# VGGnet_fast_rcnn_iter_490000.ckpt.index
	# VGGnet_fast_rcnn_iter_490000.ckpt.meta
	```

### Detections on your own images
Run the following command.

```bash
python ./tools/demo.py --model output/faster_rcnn_end2end/coco_2014_train/VGGnet_fast_rcnn_iter_490000.ckpt --img-path path_to_img_folder
```

All your annotated images will be saved in a directory called ```detections_test```

### The result of testing on coco_2014_minival 
- Tensorflow model [[Google Drive]](https://drive.google.com/file/d/0Bw0qMqgwZcafZlRqRDYxSnBkNFE/view?usp=sharing)
- Results are similar to rbgirshick's [Caffe version](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/README.md#coco-faster-r-cnn-vgg-16-trained-using-end-to-end)
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.397
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
```
