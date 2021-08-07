import numpy as np
import torch
from torchvision import transforms
from typing import Callable, Union, List, Dict, Tuple
from .common import Explanation

ceil_div = lambda x, y: (x - 1) // y + 1
#split array every d element
array_split = lambda x, d: [x[i: i + d] for i in [d*j for j in range(ceil_div(len(x), d))]]

class CausalCurve():
    def __init__(
            self,
            model: torch.nn.Module,
            step: int,
            score_fn_gen: Callable[[Explanation], Callable[[torch.Tensor], torch.Tensor]],
            substrate_fn: Callable[[torch.Tensor], torch.Tensor] = torch.zeros_like,
            stop_condition: Callable[[torch.Tensor, torch.Tensor], bool] = lambda x, y: False,
            batch_size: int = 1,):
        """ Create deletion/insertion curve instance.
        Args:
            model (nn.Module): Black-box model being explained.
            step (int): number of pixels modified per iteration.
            substrate_fn (callable): a transform from old pixels to new pixels.
            score_fn (callable): produce a scoring func mapping logits to score(s) with explanation. 
            stop_condition (callable): the condition for breaking loop,
                receiving entire curve (prefilled with nan), and new val as input.
            batch_size (int): run n modifications at once
        """
        self.model = model
        self._get_start_end_ = lambda x: (x, substrate_fn(x))
        self.step = step
        self.score_fn_gen = score_fn_gen
        self.stop_condition = stop_condition
        #self.substrate_fn = substrate_fn
        self.batch_size = batch_size

    def single_run(
            self,
            img_tensor: torch.Tensor,
            explanation: Explanation,
            output_img: bool = False):
        """run the score curve on single img explanation pair
        Args:
            img_tensor (tensor): normalized image.
            explanation (Explanation): the explanation.
            output_img (bool): whether to output the modified image at the end point
        Return:
            curve (tensor): the scores array with nan filling the escaped part
            sorted_saliency (list of tensor): the sorted saliency aligned to score
            [img_tensor_m (tensor): the modified img at the stop point.]
        """

        device = img_tensor.device
        #get scoring function
        scoring = self.score_fn_gen(explanation)

        assert len(img_tensor.shape) == 4, "image must be in shape (1, C, H, W)!"
        height, width = img_tensor.shape[-2:]
        total_len = height * width

        flat_saliency = explanation.saliency.flatten()
        assert len(flat_saliency) == total_len, "saliency shape and image not match!"
        flat_order = flat_saliency.argsort(descending = True)
        
        #allocate tasks, batchs[ chunks[ ids ] ]
        chunks = array_split(flat_order, self.step)
        batchs = array_split(chunks, self.batch_size)

        max_len = len(chunks) + 1
        # init return sorted saliency
        sorted_saliency = torch.zeros_like(flat_saliency)
        sorted_saliency[:] = np.nan
        sorted_saliency = array_split(sorted_saliency, self.step)

        #create buf for running batch
        buf = torch.zeros(self.batch_size, *img_tensor.shape[1:]).to(device)
        #save model mode
        is_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            start, end = self._get_start_end_(img_tensor)
            buf[:] = start

            logits = self.model(start)
            score = scoring(logits)
            assert len(score) == 1
            #initialize return score to nan
            curve = torch.zeros(max_len, *score.shape[1:]).to(device)
            curve[:] = np.nan
            curve[0] = score[0]

            break_flag = False
            img_tensor_m = start
            for i, b_chunks in enumerate(batchs):
                len_batch = len(b_chunks)
                #load modified images to buf
                for j in range(len_batch):
                    buf.flatten(2, 3)[j:, :, b_chunks[j]] = end.flatten(2, 3)[:, :, b_chunks[j]]
                #run
                logits = self.model(buf[:len_batch])
                score = scoring(logits)
                for k, (s, chk) in enumerate(zip(score, b_chunks)):
                    #check terminate condition
                    if self.stop_condition(curve, s):
                        break_flag = True
                        break

                    curve[1 + i*self.batch_size + k] = s
                    sorted_saliency[i*self.batch_size + k] = flat_saliency[chk]
                    img_tensor_m = buf[k]

                if break_flag:
                    break

                buf[:-1] = buf[-1]
                img_tensor_m = img_tensor_m.clone()
        if is_training:
            self.model.train()
        if output_img:
            return curve, sorted_saliency, img_tensor_m
        else:
            return curve, sorted_saliency

def get_gaussian_blur(kernel_size: int, sigma: float):
    """ Return a GaussianBlur substrate_fn
    Args:
        kernel_size (int): size of the Gaussian kernel
        sigma (float): sigma of the gaussian kernel
    Return:
        a gaussian_blur substrate_fn
    """
    return transforms.GaussianBlur(kernel_size = kernel_size, sigma = sigma)

def stop_at_val(val: float):
    """ Return a condition_fn returning True when the curve crosses val.
    Args:
        val (float): the thres
    Return:
        stop(curve), a function return True when the curve crosses val.
    """
    def stop(curve: torch.Tensor, new_val: float):
        """stop condition for when last element of curve and new val
        intersect with val
        """
        curve = curve[~torch.isnan(curve)]
        if len(curve) == 0:
            return False
        if (curve[-1] >= val).item() and (new_val <= val).item():
            return True
        if (curve[-1] <= val).item() and (new_val >= val).item():
            return True
        return False
        #mask = curve >= val
        #flag = mask.any() and ~mask.all()
        #if flag.item():
        #    print(curve, new_val)
        #return flag.item()
    return stop

def stop_at_percent(val: int):
    """ Return a condition_fn returning True when the length of curve reaches val.

    Args:
        val (int): the stop length

    Return:
        stop(curve), a function return True when the length of curve reaches val.
    """
    def stop(curve: torch.Tensor, new_val: float):
        curve = curve[~torch.isnan(curve)]
        return len(curve) >= val

    return stop

def score_gen(explain):
    """ create a function for the curve running in CausalMetric.
    Args:
        explain (Explanation): the explanation
    Return:
        the scoring function taking logits as input, output can be scalar or 1 dim tensor.
    """

    def score_fn(logits):
        probs = logits.softmax(-1)
        scores = [
                probs[:, explain.pos].sum(-1),
                probs[:, explain.neg].sum(-1)
                ]
        #contrastive score at the last col
        scores.append(scores[0]*(1-scores[1]))
        return torch.stack(scores, -1)

    return score_fn

### Metrics start
def CAUC(curve: torch.Tensor):
    """Calculate auc score along last dim
    Args:
        curve (tensor): the tensor storing the scores.
    Return:
        the auc val.
    """
    device = curve.device
    masks = curve.isnan()
    pad = torch.ones((*masks.shape[:-1], 1),).to(device)
    masks = torch.cat((1.0*masks, pad), -1)
    #truncate last dim at first nan
    lengthes = masks.argmax(-1)
    last_vals = torch.gather(curve, -1, lengthes.unsqueeze(-1) - 1)
    first_vals = torch.gather(curve, -1, torch.zeros_like(lengthes).unsqueeze(-1))
    areas = curve.nansum(-1) - 0.5*torch.cat([first_vals, last_vals], -1).nansum(-1)
    areas = areas / curve.shape[-1]
    return areas


supported_metric_map = {
        "CAUC": lambda curve, aligned_saliency: CAUC(curve),
        }



class CausalMetric(CausalCurve):
    def __init__(
            self,
            model: torch.nn.Module,
            metric,
            step: int,
            substrate_fn: Callable[[torch.Tensor], torch.Tensor] = torch.zeros_like,
            batch_size: int = 1,
            delta = 0.5,
            tau_thres=0.05,
            smooth_len=3
    ):
        """ 

        Args:
            model (nn.Module): Black-box model being explained.
            metric (str or list[str]): the metric name or list of metric name
            step (int): number of pixels modified per iteration.
            substrate_fn (callable): a transform from old pixels to new pixels.
            batch_size (int): run n modifications at once
            delta (float): the delta for stoping the loop.
            tau_thres(float): the threshold to penalize the CDROP score, where \tau=0.05*Number of all pixels in the image
            smooth_len(int): For robustness, we smooth s(n_{\delta}+1) using a default sliding window of size 3.
            CDROP(H,m|x,c,C')=\frac{s(1)-s(n_{\delta}+1)}{log_{2}(1+max\{n_{\delta},\tau\}/\tau)}

        """
        
        super().__init__(
                model,
                step, 
                substrate_fn = substrate_fn,
                score_fn_gen = score_gen,
                stop_condition = False, #replace the stop condition later
                batch_size = batch_size
                )
        
        self.metric = metric
        self.delta = delta
        self.tau_thres=tau_thres
        self.smooth_len=smooth_len
        if self.metric=='CAUC' or 'CAUC' in self.metric:
            self.metric_fn = supported_metric_map['CAUC']

    def single_run(
            self,
            img_tensor: torch.Tensor,
            explanation: Explanation,
            visual_fn: Callable[
                [torch.Tensor, List[torch.Tensor], torch.Tensor], None
                ] = None):
        """run the metric on single img explanation pair
        Args:
            img_tensor (tensor): normalized image.
            explanation (Explanation): the explanation.
        Return:
            result (tensor): the metric scores,
        """
        result={}

        #get the scoring func
        scoring = self.score_fn_gen(explanation)
        #get origin score
        is_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logits = self.model(img_tensor)
            score = scoring(logits)
            #setting stop signal

            #if delta<=1: stop at the \delta-salient pixels such that H(x)>=\delta max_x{H(x)}
            #if delta>1: regard the entered delta as the number of erased pixels

            if self.delta<=1:
                thres = explanation.saliency.min() + (explanation.saliency.max() - explanation.saliency.min()) * self.delta
                count = (1.0*(explanation.saliency > thres)).sum().item()
            if self.delta>1:
                count=self.delta

            stop_len = ceil_div(int(count), self.step)
            stop_fn = stop_at_percent(stop_len)
            #use the contrastive score at the last col as the stop signals
            self.stop_condition = lambda x, new_val: stop_fn(x[:, -1], new_val[-1])

            if visual_fn:
                #visualize result if given visual_fn
                curve, sorted_saliency, img_m = super().single_run(img_tensor, explanation, output_img = True)
                visual_fn(explanation, curve, sorted_saliency, img_m)
            else:
                curve, sorted_saliency = super().single_run(img_tensor, explanation)
        
            m_curve = curve[:, -1]

            #Calculate the CAUC if it is in the metrics 
            if self.metric=='CAUC' or 'CAUC' in self.metric:
                result['CAUC'] = self.metric_fn(m_curve, sorted_saliency)

            #Calculate the CDROP if it is in the metrics
            if self.metric=='CDROP' or 'CDROP' in self.metric:
                tau=self.tau_thres*img_tensor.shape[2]*img_tensor.shape[3]
                if stop_len>=self.smooth_len:
                    CDROP=m_curve[0]-torch.mean(m_curve[stop_len-self.smooth_len:stop_len])
                elif stop_len<self.smooth_len:
                    CDROP=m_curve[0]-m_curve[stop_len-1]
                CDROP /= np.log2(1+max(count, tau )/tau)
                result['CDROP'] =CDROP
            return result, count

from .utils import get_class_name, tensor_imshow
def visual_evaluation(
        explain: Explanation,
        curve: torch.Tensor,
        aligned_saliency: List[torch.Tensor],
        img_m: torch.Tensor,
        get_class_name: Callable[[int], str] = get_class_name):
    """ visualize evaluation processs for causal metric
    Args:
        explain (Explanation): the Explanation.
        curve (Tensor): the tensor storing the curve scores.
        aligned_saliency (list of Tensor): a list of Tensor storing values of saliency,
            corresponding to every point in curve.
        get_class_name (func): translate label id to str.
    """
    tx_size=45
    
    import matplotlib as mpl
    label_size = 40
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 

    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as mtick
    sns.set_style('darkgrid')

    ax1 = plt.subplot(221)
    # the modified img.
    #H, W = img_m.shape[-2:]
    #img_npy = np.transpose(img_m.cpu().numpy().reshape(-1, H, W), (1, 2, 0))
    tensor_imshow(img_m)
    ax1.grid(False)

    ax2 = plt.subplot(122)
    #the score
    curve = curve.cpu().numpy()
    tau = np.arange(curve.shape[0])
    tau = tau / len(tau)
    tau *= 100 #consistent with the percent formatter on x axis

    ax2.plot(tau, curve[:, 0], label = " +".join(get_class_name(l) for l in explain.pos))
    ax2.plot(tau, curve[:, 1], label = " +".join(get_class_name(l) for l in explain.neg))
    ax2.fill_between(tau, curve[:, 0], alpha = 0.2)
    ax2.fill_between(tau, curve[:, 1], alpha = 0.2)
    ax2.set_xlim(0.001, tau[~np.isnan(curve[:, 0])][-1])
    ax2.set_ylim(0)
    ax2.set_ylabel("probability",fontsize= tx_size)
    ax2.set_xlabel("pixel modified",fontsize= tx_size)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend(prop={'size': 55})

    ax3 = plt.subplot(223)
    curve = curve[:, -1]
    curve = curve[~np.isnan(curve)]
    tau = tau[:len(curve)]


    p_label = lambda labels: "P(" + " +".join(get_class_name(l) for l in labels) + ")"
    title = p_label(explain.pos) + " * (1-" + str(p_label(explain.neg))+')'
    ax3.plot(tau, curve, label = "origin", color="green")
    #ax3.plot(tau, curve_ft, label = "fitted")
    ax3.fill_between(tau, curve, alpha = 0.1,color="green")
    ax3.set_title(title,fontsize= tx_size)
    ax3.set_xlim(0.001, tau.max())
    ax3.set_ylim(0)
    ax3.set_ylabel("contrastive score",fontsize= tx_size)
    ax3.set_xlabel("pixel modified",fontsize= tx_size)
    ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
        
        
