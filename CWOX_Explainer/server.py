from flask import Flask, session, send_from_directory, redirect, request
from flask_session import Session
import json, base64, io
from PIL import Image
import numpy as np

import torch, torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"
as_numpy = lambda x: x.detach().cpu().numpy()
as_tensor = lambda x: torch.tensor(x).to(device)

with open("./site/imagenet_class_index.json") as f:
    indx2label = json.load(f)

with open("./site/support_info.json") as f:
    support_info = json.load(f)
    supported_methods = {method["value"]: method for method in support_info["methods"]}
    supported_models = {model["value"]: model for model in support_info["models"]}


def decode_predictions(preds, k=200):
    # return the top k results in the predictions
    return [
        [
            (*indx2label[str(i)], i.item(), pred[i].item())
            for i in pred.argsort()[::-1][:k]
        ]
        for pred in as_numpy(preds)
    ]

image_shape = [224, 224]

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

get_tree = lambda x: [
    {"name": name, "children": get_tree(children)}
    for name, children in x.named_children()
]


def load_model(name):
    assert name in supported_models.keys()
    model = getattr(torchvision.models, name)(pretrained=True)
    model.to(device)
    model.eval()
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

from grad_cam_mwp_cwox import grad_cam_cwox, mwp_cwox
from rise_cwox import rise_cwox
from lime_cwox import lime_cwox
def explain_cwox(x, model, method, clusters, para):
    assert method in supported_methods
    if method == "grad_cam":
        res = grad_cam_cwox(x, model, clusters, para["cluster_layer"], para["class_layer"])
    elif method == "mwp":
        res = mwp_cwox(x, model, clusters, para["cluster_layer"], para["class_layer"])
    elif method == "rise":
        s_cluster, s_class = int(para["Mask_Size_cluster"]), int(para["Mask_Size_class"])
        p_cluster, p_class = float(para["Mask_Prob_cluster"]), float(para["Mask_Prob_class"])
        res = rise_cwox(x, model, clusters, s_cluster, p_cluster, s_class, p_class)
    elif method == "lime":
        ker_size_cluster = int(para["Kernel_Size_cluster"])
        res = lime_cwox(x, model, clusters, ker_size_cluster)
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
        s, p = int(para["Mask_Size"]), float(para["Mask_Prob"])
        res = rise_swox(x, model, labels, s, p)
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
        x = transform(image)[None]
        x = x.to(device)
        session["target_input"] = x

        model_name = request.form["model"]
        assert model_name in supported_models.keys()
        model = load_model(model_name)
        session["model"] = model
        session["model_name"] = model_name
        #session["modules"] = modules

        logits = model(x)
        preds = logits.softmax(-1)
        c_perp = (-preds * preds.log()).sum().exp().item()
        results = decode_predictions(preds)[0]
        return json.dumps({"preds": results, "c_perp": c_perp})

    @app.route("/explain", methods=["POST"])
    def explain_handler():
        clusters = request.form["clusters"]
        clusters = [
            [int(label.replace("c", "")) for label in cluster]
            for cluster in json.loads(clusters)
        ]
        model_name = session["model_name"]
        model = session["model"]
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
        print(para)

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
