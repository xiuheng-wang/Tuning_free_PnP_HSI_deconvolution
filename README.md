# Tuning-free Plug-and-Play Hyperspectral Image Deconvolution with Deep Priors

Steps for the public CAVE dataset:

1. Run cave_processing.py to process the data;

2. Run blurring_image.m to blurring the raw hyperspectral images;

3. Run main.sh to test our method.

For our real-world dataset (./data/Hide/):

Simply run main_real.sh to test our method.

The trained parameters of the B3DDN is stored in ./models/hsidb_epoch500.pkl, if you want to train the B3DDN by yourself:

1. Run cave_processing.py to process the data;

2. Run train.py.

For any questions, feel free to email me at xiuheng.wang@oca.eu.

If this code is helpful for you, please cite our paper as follows:

    @article{wang2022tuning,
      title={Tuning-free Plug-and-Play Hyperspectral Image Deconvolution with Deep Priors},
      author={Wang, Xiuheng and Chen, Jie and Richard, C{\'e}dric},
      journal={arXiv preprint arXiv:2211.15307},
      year={2022}
    }
    @inproceedings{wang2020learning,
      title={Learning Spectral-Spatial Prior Via 3DDNCNN for Hyperspectral Image Deconvolution},
      author={Wang, Xiuheng and Chen, Jie and Richard, C{\'e}dric and Brie, David},
      booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={2403--2407},
      year={2020},
      organization={IEEE}
    }

