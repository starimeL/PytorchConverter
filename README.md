# Pytorch Converter
Pytorch model to Caffe &amp; [ncnn](https://github.com/Tencent/ncnn)

## Model Examples
  - SqueezeNet from torchvision
  - [ResNet50] (to be loaded) (with ceiling_mode=True)
  - AnimeGAN pretrained model from author (https://github.com/jayleicn/animeGAN)
  - UNet (no pretrained model yet, just default initialization)
        
## Attentions
  - **Mind the difference on ceil_mode of pooling layer among Pytorch and Caffe, ncnn**
    - You can convert Pytorch models with all pooling layer's ceil_mode=True.
    - Or compile a custom version of Caffe/ncnn with floor() replaced by ceil() in pooling layer inference.

  - **Python Error: AttributeError: grad_fn**
    - Update your version to pytorch-0.2.0 and torchvision-0.1.9 at least.

  - **Other Python packages requirements:**
    - to Caffe: numpy, protobuf (to gen caffe proto)
    - to ncnn: numpy
    - for testing Caffe result: pycaffe, cv2
