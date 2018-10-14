import sys
import json
import argparse
sys.path.append('../coco-caption')
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

parser = argparse.ArgumentParser(description='argument parser')
parser.add_argument('-hyp', '--hypo_path', type=str, default=None,
                    help='path for test output json file')
parser.add_argument('-ref', '--ref_path', type=str, default='./data/annotations/captions_val2014.json',
                    help='path for test output json file')

args = parser.parse_args()

annFile = args.ref_path
resFile = args.hypo_path

print('reference: ', annFile)
print('hypothesis: ', resFile)


coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()

print('\n\n-----results-----')
for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))
