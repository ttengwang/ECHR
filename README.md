# Event-Centric Hierarchical Representation for Dense Video Captioning
In this [paper](https://ieeexplore.ieee.org/document/9160989), we propose event-centric hierarchical representation for dense video captioning. We enhance the event-level representation by capturing rich relationship between events in terms of both temporal structure and semantic meaning. Then, a caption generator with late fusion is developed to generate surrounding-event-aware and topic-aware sentences, conditioned on the hierarchical representation of visual cues from the scene level, the event level, and the frame level.

This repo contains main codes of experiments on the ActivityNet Captions dataset.

# Usage
- Install `Python 2.7 + PyTorch 0.4 + CUDA 10.0`. Then run `pip install environment.txt`.
- Prepare the video and annotation data. Please refer to [url](https://drive.google.com/drive/folders/1GYnmyXNHSGZCZyxUF7xK6zqGCM7yIHT4?usp=sharing).
- Training scripts are in this folder `experiments`.
# Reference
```
@ARTICLE{Wang2020echr,
  author={T. {Wang} and H. {Zheng} and M. {Yu} and Q. {Tian} and H. {Hu}},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Event-Centric Hierarchical Representation for Dense Video Captioning}, 
  year={2020}}
```
# Acknowledgement
This code is based on [ImageCaptioning.Pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch).