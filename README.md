# Pytorch Converter
Pytorch model to Caffe &amp; [ncnn](https://github.com/Tencent/ncnn)

## Model Examples
  - SqueezeNet from torchvision
  - DenseNet from torchvision
  - [ResNet50](https://drive.google.com/file/d/0B5B31rlbCRZfcS1rY3BtVWhDREk/view?usp=sharing) (with ceiling_mode=True)
  - MobileNet
  - AnimeGAN pretrained model from author (https://github.com/jayleicn/animeGAN)
  - SSD-like object detection net(for ncnn)
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

  - **Model Loading Error**
    - Use compatible model saving & loading method, e.g.    

      ```
      # Saving, notice the difference on DataParallel
      net_for_saving = net.module if use_nn_DataParallel else net
      torch.save(net_for_saving.state_dict(), path)
      
      # Loading
      net.load_state_dict(torch.load(path, map_location=lambda storge, loc: storage))
      ```
