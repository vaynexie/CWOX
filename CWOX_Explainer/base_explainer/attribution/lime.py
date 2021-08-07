import os, json
import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb
from skimage.segmentation import felzenszwalb, slic, quickshift
from PIL import Image
import scipy as sp
import sys
import inspect
import types
import copy
from functools import partial
from tqdm.auto import tqdm
import types


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



def get_input_tensors(img):
    def get_input_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])       
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])    
        return transf
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224)
    ])    
    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    
    return transf    


def batch_predict(model,images):
    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def has_arg(fn, arg_name):
    """Checks if a callable accepts a given keyword argument.
    Args:
        fn: callable to inspect
        arg_name: string, keyword argument name to check
    Returns:
        bool, whether `fn` accepts a `arg_name` keyword argument.
    """
    if sys.version_info < (3,):
        if isinstance(fn, types.FunctionType) or isinstance(fn, types.MethodType):
            arg_spec = inspect.getargspec(fn)
        else:
            try:
                arg_spec = inspect.getargspec(fn.__call__)
            except AttributeError:
                return False
        return (arg_name in arg_spec.args)
    elif sys.version_info < (3, 6):
        arg_spec = inspect.getfullargspec(fn)
        return (arg_name in arg_spec.args or
                arg_name in arg_spec.kwonlyargs)
    else:
        try:
            signature = inspect.signature(fn)
        except ValueError:
            # handling Cython
            signature = inspect.signature(fn.__call__)
        parameter = signature.parameters.get(arg_name)
        if parameter is None:
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))

class BaseWrapper(object):
    def __init__(self, target_fn=None, **target_params):
        self.target_fn = target_fn
        self.target_params = target_params
    def _check_params(self, parameters):
        a_valid_fn = []
        if self.target_fn is None:
            if callable(self):
                a_valid_fn.append(self.__call__)
            else:
                raise TypeError('invalid argument: tested object is not callable,\
                 please provide a valid target_fn')
        elif isinstance(self.target_fn, types.FunctionType) \
                or isinstance(self.target_fn, types.MethodType):
            a_valid_fn.append(self.target_fn)
        else:
            a_valid_fn.append(self.target_fn.__call__)
        if not isinstance(parameters, str):
            for p in parameters:
                for fn in a_valid_fn:
                    if has_arg(fn, p):
                        pass
                    else:
                        raise ValueError('{} is not a valid parameter'.format(p))
        else:
            raise TypeError('invalid argument: list or dictionnary expected')
    def set_params(self, **params):
        self._check_params(params)
        self.target_params = params
    def filter_params(self, fn, override=None):
        override = override or {}
        result = {}
        for name, value in self.target_params.items():
            if has_arg(fn, name):
                result.update({name: value})
        result.update(override)
        return result


class SegmentationAlgorithm(BaseWrapper):
    def __init__(self, algo_type, **target_params):
        self.algo_type = algo_type
        #print(self.algo_type)
        if (self.algo_type == 'quickshift'):
            BaseWrapper.__init__(self, quickshift, **target_params)
            kwargs = self.filter_params(quickshift)
            self.set_params(**kwargs)
        elif (self.algo_type == 'felzenszwalb'):
            BaseWrapper.__init__(self, felzenszwalb, **target_params)
            kwargs = self.filter_params(felzenszwalb)
            self.set_params(**kwargs)
        elif (self.algo_type == 'slic'):
            BaseWrapper.__init__(self, slic, **target_params)
            kwargs = self.filter_params(slic)
            self.set_params(**kwargs)
    def __call__(self, *args):
        return self.target_fn(args[0], **self.target_params)


class ImageExplanation(object):
    def __init__(self, image, segments):
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None
    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        #print(exp)
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs
    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)
    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)
            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)
    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   pos_labels,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        weights = self.kernel_fn(distances)        
        if len(pos_labels)>1:
            pos_column= neighborhood_labels[:, pos_labels[0]]
            for pp in pos_labels[1:]:pos_column+=neighborhood_labels[:, pp]
            pos_column/=len(pos_labels)
        if len(pos_labels)==1: pos_column= neighborhood_labels[:, pos_labels[0]]
        labels_column=pos_column  
        kkk=0
        for ll in labels_column:
            if ll<0:kkk+=1
        labels_column1=[]
        neighborhood_data1=[]
        weights1=[]
        for ii in range(len(labels_column)):
            if labels_column[ii]>0:
                labels_column1.append(labels_column[ii])
                neighborhood_data1.append(neighborhood_data[ii])
                weights1.append(weights[ii])
        labels_column=np.array(labels_column1)
        neighborhood_data=np.array(neighborhood_data1)
        weights=np.array(weights1)                                              
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)
        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: x[1], reverse=True),
                prediction_score, local_pred)

class LimeImageExplainer(object):
    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        kernel_width = float(kernel_width)
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)
    def explain_instance(self, image, model,classifier_fn, segments,pos_labels=None,neg_labels=None,
                         hide_color=None,
                         num_features=100000, num_samples=1000,
                         batch_size=30,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)
        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color
        data, labels = self.data_labels(image, model,fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=False)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()
        ret_exp = ImageExplanation(image, segments)
        (ret_exp.intercept['cwox1_pos'],ret_exp.local_exp['cwox1_pos'],ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, pos_labels, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        k=0
        for neg_labels1 in neg_labels:
            name1='cwox1_neg_'+str(k)
            (ret_exp.intercept[name1],ret_exp.local_exp[name1],ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
            data, labels, distances, neg_labels1, num_features,
            model_regressor=model_regressor,
            feature_selection=self.feature_selection)
            k+=1      
        data, labels = self.data_labels(image,model, fudged_image, segments,
                            classifier_fn, num_samples,
                            batch_size=batch_size,
                            progress_bar=False)
        distances = sklearn.metrics.pairwise_distances(
        data,
        data[0].reshape(1, -1),
        metric=distance_metric).ravel()
        for pp in pos_labels:
            name1='cwox'+str(pp)
            #print(name1)
            pos_labels1=(pp,)
            (ret_exp.intercept[name1],ret_exp.local_exp[name1],ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    data, labels, distances, pos_labels1, num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)
        return ret_exp
    def data_labels(self,
                    image,
                    model,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=30,
                    progress_bar=True):
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(model,np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(model,np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)
    

def output_exp_final(exp_pos,exp_neg):
    exp1=[]
    for i in exp_pos:
        if i[1]>0:exp1.append(i[1])
    temp_max=np.max(exp1)
    temp_min=np.min(exp1)
    exp_pos1=[]
    for i in exp_pos:
        if i[1]>0:exp_pos1.append((i[0],(i[1]-temp_min)/(temp_max-temp_min)))
        if i[1]<0:exp_pos1.append((i[0],0))
    exp1=[]
    for i in exp_neg:
        if i[1]>0:exp1.append(i[1])

    temp_max=np.max(exp1)
    temp_min=np.min(exp1)
    exp_neg1=[]
    for i in exp_neg:
        if i[1]>0:exp_neg1.append((i[0],(i[1]-temp_min)/(temp_max-temp_min)))
        if i[1]<0:exp_neg1.append((i[0],0))
    exp_pos_dict={}
    exp_neg_dict={}

    for i in exp_pos1:
        exp_pos_dict[i[0]]=i[1]
    for i in exp_neg1:
        exp_neg_dict[i[0]]=i[1]

    exp_final=[]
    for i in exp_pos_dict:
        if i in exp_neg_dict:
            exp_final.append((i,exp_pos_dict[i]-exp_neg_dict[i]))
    return exp_final

def output_exp_final_final(exp_pos,exp_neg_list):
    exp_final_final_1=[]
    d={}
    for kkk in exp_neg_list:
        exp_final=output_exp_final(exp_pos,kkk)
        exp_final_final_1.extend(exp_final)
    for (word,score) in exp_final_final_1:
        if word in d:
            d[word]=d[word]+score
        else:
            d[word]=d.setdefault(word,0)+score
    exp_final_final= [(k, v) for k, v in d.items()] 
    exp_final_final.sort(key = lambda x: x[1],reverse=True)  
    return exp_final_final

def times_support(cwox1,cwox2,delta=60):
    cwox1_sort=sorted(cwox1,key=lambda x: x[0])
    cwox1_value=[]
    for i in cwox1_sort:
        cwox1_value.append(i[1])
    support_value=np.percentile(cwox1_value,delta)
    cwox1_support_dict={}
    for i in cwox1_sort:
        if i[1]>support_value:
            cwox1_support_dict[i[0]]=1
        else:
            cwox1_support_dict[i[0]]=0
    cwox2_sort=sorted(cwox2,key=lambda x: x[0])
    cwox2_support=[]
    for i in cwox2_sort:
        if i[0] in cwox1_support_dict:
            cwox2_support.append((i[0],i[1]*cwox1_support_dict[i[0]]))
        
    cwox2_support.sort(key = lambda x: x[1],reverse=True)  
    return cwox2_support


def get_image_and_mask1(image,segments, exp, positive_only=True, negative_only=False, hide_rest=False,
                       num_features=5, min_weight=0.):


    exp=sorted(exp,key=lambda x: x[1],reverse=True)
    mask = np.zeros(segments.shape, segments.dtype)
    if hide_rest:
        temp = np.zeros(image.shape)
    else:
        temp = image.copy()
    if positive_only:
        fs = [x[0] for x in exp
              if x[1] > 0 and x[1] > min_weight][:num_features]
        
        #print(fs)
    if negative_only:
        fs = [x[0] for x in exp
              if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
    if positive_only or negative_only:
        for f in fs:
            temp[segments == f] = image[segments == f].copy()
            mask[segments == f] = 1
        return temp, mask
    else:
        for f, w in exp[:num_features]:
            if np.abs(w) < min_weight:
                continue
            c = 0 if w < 0 else 1
            mask[segments == f] = -1 if w < 0 else 1
            temp[segments == f] = image[segments == f].copy()
            temp[segments == f, c] = np.max(image)
        return temp, mask
    
def sal_map_to_boundry(sal_map,kk,a=True,b=False):
    temp, mask = get_image_and_mask1(explanation.image,explanation.segments,sal_map,positive_only=a, num_features=kk,hide_rest=b)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    return img_boundry1
