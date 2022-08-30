## [Abstract]

![](https://images.velog.io/images/heaseo/post/000c6fe2-227b-4ff3-930f-83a09fec26b3/Real-ESRGAN%20degradation.png)

<br>

## Requirements
``` bash
- pip3 install pytorch
- pip3 install imgaug
- pip3 install tensorboard
- pip3 install scipy
- pip3 install opencv-python
```

## Train
``` bash
python3 train.py --train-file ${train_datasets} --eval-file ${valid_datasets} --outputs-dir ${save_model_dir} --scale ${2 or 4} --resume-net ${BSRNet.pth}
```

## Test
``` bash
python3 test.py --weights-file ${BSRGAN.pth} --image-file ${image file path} --scale ${2 or 4}
```

<br>

## Results
![](https://images.velog.io/images/heaseo/post/1808c539-6646-4ca5-aede-7a3c08affd12/teaser.jpg)# Real-ESRGAN
