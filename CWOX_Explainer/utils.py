import tempfile
from PIL import Image
import numpy as np

def to_PilTempPath(img):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as fp:
        img = img.asnumpy()
        assert len(img.shape) == 3
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i, (m, s) in enumerate(zip(mean, std)):
            img[i] *= s
            img[i] += m
        img = np.transpose(img, [1, 2, 0])
        img = (255*img).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(img)
        image.save(fp, format="png")
    return fp.name
