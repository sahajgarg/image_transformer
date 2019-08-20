# Image Transformer (pytorch)

A Pytorch implementation of the [Image Transformer](https://arxiv.org/abs/1802.05751). Code adapted from the official implementation in the [tensor2tensor](https://github.com/tensorflow/tensor2tensor/) library. 

Currently supports unconditional image generation for CIFAR10, where the distribution for a pixel can either be categorical or discretized mixture of logistics (as in PixelCNN++). Supports block-wise attention using Local 1D blocks, which perform the best in evaluations on CIFAR10. 

Pull requests are welcome for supporting Local 2D blocked attention, image to image superresolution, or class conditional generation! 

### Running the code

Install the requirements with `pip install -r requirements.txt`. Then, run the code, with the optional sample flag to generate samples during train time.

```
python3 train_transformer.py --doc run_name --config transformer_cat.yml --sample
```
