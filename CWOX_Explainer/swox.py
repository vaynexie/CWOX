import mindspore as ms
import numpy as np

from mindspore.explainer.explanation import RISE_quick, GradCAM, ExcitationBackProp, LIME

def rise_swox(img, model, labels, size1=10, p1=0.1):
    # Load black box model for explanations
    #Generate Mask for cluster
    explainer = RISE_quick(model, 5000, p1, size1)
    sal = explainer(img, labels)
    print(sal)
    return {"label_{}".format(i): sal[i].asnumpy() for i in labels}

def grad_cam_swox(img, model, labels, class_layer):
    grad_cam = GradCAM(model, class_layer)
    return {
            "label_{}".format(i): grad_cam(img, i).asnumpy()[0, 0]
            for i in labels
            }

def mwp_swox(img, model, labels, class_layer):
    return {
            "label_{}".format(i): ExcitationBackProp(model, class_layer)(img, i).asnumpy()[0, 0]
            for i in labels
            }

from utils import to_PilTempPath
import os
def lime_swox(img, model, labels, kernel_size1 = 5, number_sample1 = 2500):
    path = to_PilTempPath(img[0])
    lime = LIME(model, kernel_size1, number_sample1)
    sal = lime(path, labels)
    os.remove(path)
    return {"label_{}".format(i): s for i, s in zip(labels, sal)}
