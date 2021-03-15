

Please contact yufanzho@buffalo.edu if you have any question.

## Datasets

Please revise the dataset path in main.py.

Links for datasets:
```
cifar-10: wget --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
stl-10: python download_stl.py
ImageNet: http://image-net.org/challenges/LSVRC/2012
CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
```
For datasets other than cifar-10 (stl-10, ImageNet, CelebA), need to run build_records.py or build_imgnet_dataset.py first before training to build tf_record files.

## Example

Run

```
python main.py -gpu 0 -config_file w2flow_cifar10_dk.yml
```

Generated samples on CIFAR-10
<p align="center">
	<img src="https://github.com/drboog/Heat-Kernel/blob/main/samples/cifar10.png">
</p>



## Acknowledgement

The implementation is based on this repository: https://github.com/MichaelArbel/Scaled-MMD-GAN.
