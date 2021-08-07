from flask import Flask, session, send_from_directory, redirect, request
from flask_session import Session
import json, base64, io
from PIL import Image
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
import mindspore.ops.functional as F

Tensor = ms.Tensor
exp_ms = P.Exp()
sum_ms = P.ReduceSum()
log_ms = P.Log()


with open("./site/imagenet_class_index.json") as f:
    indx2label = json.load(f)

with open("./site/support_info.json") as f:
    support_info = json.load(f)
    supported_methods = {method["value"]: method for method in support_info["methods"]}
    supported_models = {model["value"]: model for model in support_info["models"]}

import mindspore.nn as nn
softmax = nn.Softmax()
def decode_predictions(preds, k=200):
    # return the top k results in the predictions
    return [
        [
            (*indx2label[str(i)], i.item(), pred[i].item())
            for i in pred.argsort()[::-1][:k]
        ]
        for pred in preds.asnumpy()
    ]

image_shape = [224, 224]

import mindspore.dataset.vision.c_transforms as C
def transform(img, size):
    img = np.array(img.resize(size, Image.BILINEAR))
    img = img / 255
    img = C.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    img= C.HWC2CHW()(img)
    img =img[np.newaxis,:]
    return Tensor(img)

get_tree = lambda x: [
    {"name": name, "children": get_tree(children)}
    for name, children in x.named_children()
]

from mindspore.explainer.explanation._attribution._CWOX.imagenet_model.resnet50 import resnet_imagenet
from mindspore.explainer.explanation._attribution._CWOX.imagenet_model.googlenet import googlenet_imagenet
model_gen = {
        "resnet50": resnet_imagenet,
        "googlenet": googlenet_imagenet
        }

def load_model(name):
    assert name in supported_models.keys()
    model_fn = model_gen[name]
    model = model_fn()
    #modules = {"name": name, "children": get_tree(model)}
    return model


encode_array = lambda x: base64.encodebytes(x.astype(np.float32).tobytes()).decode('utf-8')


def cwox_explanation_stringify(explanation, other_info = {}):
    result = [
        {
            "id": c_name,
            "array": {"shape": c_info.shape, "bytes": encode_array(c_info)},
            "childrens": [
                {
                    "id": name,
                    "array": {"shape": info.shape, "bytes": encode_array(info)},
                }
                for name, info in c_mem
            ],
        }
        for c_name, c_info, c_mem in explanation
    ]
    result = {"result": result}
    result.update(other_info)
    return json.dumps(result)

def swox_explanation_stringify(explanation, other_info = {}):
    result = [
        {
            "id": name,
            "array": {"shape": info.shape, "bytes": encode_array(info)},
        }
        for name, info in explanation.items()
    ]
    result = {"result": result}
    result.update(other_info)
    return json.dumps(result)

def normalize(x):
    x /= x.amax([-1, -2], keepdim = True)
    x[x.isnan()] = 0
    return x

from cwox import grad_cam, mwp, rise, lime
def explain_cwox(x, model, method, clusters, para):
    assert method in supported_methods
    if method == "grad_cam":
        res = grad_cam(x, model, clusters, para["cluster_layer"], para["class_layer"])
    elif method == "mwp":
        res = mwp(x, model, clusters, para["cluster_layer"], para["class_layer"])
    elif method == "rise":
        res = rise(x, model, clusters)
    elif method == "lime":
        res = lime(x, model, clusters)
    else:
        assert False
    explanation = [
            ("cluster_{}".format(i), res["cluster{}".format(i+1)], [(v, res.get("cluster{}_{}".format(i+1, v), np.zeros((5,5)))) for v in cluster])
            for i, cluster in enumerate(clusters)
            ]
    return explanation

from swox import grad_cam_swox, mwp_swox, lime_swox, rise_swox
def explain_swox(x, model, method, labels, para):
    assert method in supported_methods
    if method == "grad_cam":
        res = grad_cam_swox(x, model, labels, para["layer"])
    elif method == "mwp":
        res = mwp_swox(x, model, labels, para["layer"])
    elif method == "rise":
        S, P = int(para["Mask_Size"]), float(para["Mask_Prob"])
        res = rise_swox(x, model, labels, S, P)
    elif method == "lime":
        ker_size = int(para["Kernel_Size"])
        res = lime_swox(x, model, labels, ker_size)
    else:
        assert False
    return res


    

def create_app():
    app = Flask(__name__)
    app.secret_key = b"\r7\x8bQ\xfe\xcef\xb1^\xb4\xf0A\x83\x053t"

    app.config.from_object(__name__)
    app.config["SESSION_TYPE"] = "filesystem"
    Session(app)

    @app.route("/")
    @app.route("/site")
    def index_handler():
        response = redirect("/site/index.html")
        return response

    @app.route("/site/<path:path>")
    def static_content(path):
        response = send_from_directory("site", path)
        return response

    @app.route("/upload", methods=["POST"])
    def upload_handler():
        image = request.files["image"]

        image = Image.open(image).convert("RGB")
        x = transform(image, image_shape)
        session["target_input"] = x

        model_name = request.form["model"]
        assert model_name in supported_models.keys()
        model = load_model(model_name)
        session["model_name"] = model_name
        #session["modules"] = modules

        logits = model(x)
        preds = softmax(logits)
        results = decode_predictions(preds)[0]
        c_perp = exp_ms(sum_ms(-preds * log_ms(preds))).asnumpy().item()
        print(c_perp)
        return json.dumps({"preds": results, "c_perp": c_perp})

    @app.route("/explain", methods=["POST"])
    def explain_handler():
        clusters = request.form["clusters"]
        clusters = [
            [int(label.replace("c", "")) for label in cluster]
            for cluster in json.loads(clusters)
        ]
        model_name = session["model_name"]
        model = load_model(model_name)
        x = session["target_input"]
        method = request.form["saliency_map"]
        explain_type = request.form["type"]
        assert method in supported_methods.keys()
        assert explain_type in ["cwox", "swox"]
        para_cluster_default = {
                para["name"]:para["default"][model_name]
                for para in supported_methods[method]["parameters_cluster"]
                }
        para_class_default = {
                para["name"]:para["default"][model_name]
                for para in supported_methods[method]["parameters_class"]
                }
        para_nat_default = {
                para["name"]:para["default"][model_name]
                for para in supported_methods[method]["parameters_nat"]
                }

        para = {**para_cluster_default, **para_class_default, **para_nat_default}
        para.update(json.loads(request.form["saliency_parameters"]))

        if explain_type == "cwox":
            explanation = explain_cwox(
                x, model, method, clusters, para
            )
            attach_info = {
                    "meta": {
                        "method": method,
                        "model": model_name,
                        "para_json": json.dumps(para),
                        }
                    }
            result = cwox_explanation_stringify(explanation, attach_info)
        elif explain_type == "swox":
            attach_info = {
                    "meta": {
                        "method": method,
                        "model": model_name,
                        "para_json": json.dumps(para),
                        }
                    }
            explanation = explain_swox(
                x, model, method, sum(clusters, []), para
            )
            result = swox_explanation_stringify(explanation, attach_info)
        else:
            assert False
        return result

    return app


app = create_app()
