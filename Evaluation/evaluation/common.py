import torch
from typing import Callable, Union, List, Dict, Tuple

class Explanation():
    def __init__(
            self,
            target: Union[int, Union[List[List[int]], Tuple[List[int]]]],
            saliency: torch.Tensor,
            ):
        assert len(target) == 2, "incorrect length of target in contrast mode!"
        self.pos, self.neg = target
        self.saliency = saliency

