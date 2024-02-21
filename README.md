## Fine-Tuning for Bayer Demosaicking through Periodic-Consistent Self-Supervised Learning
This is the code for the paper _Fine-Tuning for Bayer Demosaicking through Periodic-Consistent Self-Supervised Learning_

Abstract: *Deep learning-based Bayer demosaicking methods have achieved superior performance. They require a large amount of paired images, which are not easy to collect. So most of these methods resort to using simulated images, where the raw images are sampled from the RGB images with the Bayer color filter array (CFA). However, there is a domain gap between simulated images and real images, which makes the learned networks less practical in the real world. To address this problem, we propose a simple and effective
self-supervised learning framework for Bayer demosaicking. We observe that the Bayer CFA is periodic, and its four equivalent patterns can be transformed into each other through simple translation. Accordingly, we propose the concept of periodic-consistent demosaicking, which means that a demosaicking network should produce an identical demosaicked image for the same scene captured by the four equivalent patterns. Our framework can train or fine-tune existing demosaicking networks using only single raw images. We demonstrate that our framework can fine-tune a demosaicking network to a specific camera and significantly improve the performance of existing demosaicking methods on both clean and noisy images. Source code will be released along with paper publication.*

## Requirements
* 1&ndash;8 high-end NVIDIA GPUs.


## Getting started

### Preparing datasets
We used [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset for pre-training and [Gehler_Shi](https://www2.cs.sfu.ca/~colour/data/shi_gehler/) for fine-tuning.

### Cropping images
Because it is required that all the input images have the same width and height, we recommend that you could use `crop.py` to crop the input images, the cropped images will be in a new folder called `cropped` in the folder where the original images are.

If you would like to crop your images, you should run the following command

```
python crop.py --imgs_dir [your dataset path] --width [width] --height [height]
```

### Pre-Training
If you would like to pre-train your model from scratch, you should run the following command

```
python pre_train.py --train_dir [your training dataset path] --val_dir [your validating dataset path] --batch_size 16 --epoch 50
```

The trained model checkpoint will be saved as `checkpoints/best_pretrain_unet.pth`

### Fine-Tuning
If you would like to fine-tune your model, you should run the following command

```
python fine_tune.py --train_dir [your training dataset path] --test_dir [your test dataset path] \
    --checkpoint [your checkpoint path] --batch_size [batch_size] --epoch [epoch] \
    --freeze [the parameters percentage you want to freeze] \
    --loss_pattern_num [the number of patterns that you would like to use in fine-tuning, default is set to 3] \
    --save_results [True or False]
```

The fine-tuned model checkpoint will be saved as `checkpoints/fine_tuned.pth`