# PyTorch Implementation of ResNet-preact

## Usage

```
$ python main.py --block_type basic --depth 110 --outdir results
```

### Use PyramidNet-like Residual Unit

```
$ python main.py --block_type basic --depth 110 --remove_first_relu True --add_last_bn True --outdir results
```

## References

* He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. [arXiv:1512.03385]( https://arxiv.org/abs/1512.03385 )
* He, Kaiming, et al. "Identity mappings in deep residual networks." European Conference on Computer Vision. Springer International Publishing, 2016. [arXiv:1603.05027]( https://arxiv.org/abs/1603.05027 ), [Torch implementation]( https://github.com/KaimingHe/resnet-1k-layers )


