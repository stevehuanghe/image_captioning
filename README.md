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

## Results

The models were run on a NVIDIA TITAN X Pascal GPU with 12GB memory. 

|        | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | Rouge | CIDEr | SPICE | Train epoch |
|:------:|:------:|:------:|:------:|:------:|:------:|:-----:|:-----:|:-----:|:-----:|
| NIC    |66.6    | 46.1   |  32.9  |  24.6  |-       | -     | -     | -     | -     |   
|NIC re  | 67.9   |  50.1  |35.7    |  25.3  |   23.2 |  50.1 | 80.4  |  16   |   10  |
|NIC re (bs)| 66.4|48.9| 35.8| 26.2 | 24.2|50.7|  83  |  16.7 |10|
|SA | 70.7 | 49.2 | 34.4 | 24.3 | 23.9 | -  | - | - | -  |
|SA re| 64.1 | 45.1 | 30.4 | 20.6 | 20.4 | 46.6 | 63.9 | 13.1 | 10 |
|SCA-CNN | 71.9 | 54.8 |41.1 |31.1  | 25   | -   | -  | -  | -    |
|SCA-CNN re| 57.2 |37.6 | 24.1 | 15.8 | 17.5 | 42.1 | 47.3 | 10.1 | 10|

For NIC, since I use a better feature extractor (ResNet152) than the paper [3] (GoogLeNet), my results are better than those reported in the paper. Also, beam-search can slightly impove the model's performance.

For soft-attention (SA), I did not train the model long enough, so its performance is slightly worse than the scores reported in [2].

For SCA-CNN, my results are much worse than the original results in all the evaluation metrics, which is because of the limitation of GPU memory. For one image, its visual feature is of size 2048*7*7, and the original hidden dimension of LSTM in [1] is 1000. Hence, the size of the weight matrix which is used to transfer image features as LSTM input is around 400MB. When we unroll the LSTM for back-propagation, the model needs to store both gradients and errors for this matrix in each time-step (the maximum time-step is 20). Therefore, the peak memory cost is 400MB * 20 * 2 = 16GB. Since I only have one GPU with 12 GB memory for the experiments, I could not run this model in the original setting successfully. Instead I changed the hidden dimension of LSTM from 1000 to 300, which makes the performance much worse than the original paper. In my experiments, I could only afford to run with a really small batch size of 20. 


## References

[1]	Chen, L. et al. 2017. SCA-CNN - Spatial and Channel-Wise Attention in Convolutional Networks for Image Captioning. CVPR. (2017), 6298–6306.

[2] Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International conference on machine learning. 2015.

[3] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
