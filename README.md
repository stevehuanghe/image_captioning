# Image Captioning Models in PyTorch

Some code is borrowed from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

Here are the implementations of Google-NIC[3], soft-attention[2] and SCA-CNN[1] with PyTorch and Python3.

For SCA-CNN, I do not have time to implement multi-layer attention, so I just use output of the last layer of resnet152 as 
image features.

## Usage:

### Download the repositories:

```bash
# download and follow the setup steps in coco-caption-py3.git/README.md
$ git clone https://github.com/stevehuanghe/coco-caption-py3.git
# download this respository
$ git clone https://github.com/stevehuanghe/image_captioning.git
$ cd ./image_captioning/
```
### Download and process the data

```bash
$ pip install -r requirements.txt
$ chmod +x download.sh
$ ./download.sh
$ python build_vocab.py   
$ python resize.py
```
### train the model

```bash
# for Google-NIC
$ sh train-nic.sh
# for soft-attention
$ sh train-ssa.sh
# for SCA-CNN
$ sh train-scacnn.sh
```
### generate captions using trained model
```bash
# for all configurations, please look into utils/config.py
# for Google-NIC
$ sh test-nic.sh
# for soft-attention
$ sh test-ssa.sh
# for SCA-CNN
$ sh test-scacnn.sh
```
### calculate scores for generated captions
```bash
python evaluate.py -hypo PATH_TO_SAVED_CAPTION_FILE.json
```
### manual inspection of captions and iamges

please use the visualize_caption.ipynb


[1]	Chen, L. et al. 2017. SCA-CNN - Spatial and Channel-Wise Attention in Convolutional Networks for Image Captioning. CVPR. (2017), 6298–6306.

[2] Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International conference on machine learning. 2015.

[3] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
