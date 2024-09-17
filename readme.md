## An implementation

Implementation of LpTTA in Pytorch. It is alabel propagation-based
test-time adaptation model that leverages historical outputs via
label propagation to handle temporal correlations. The main model logic is implemented in lptta.py

## Install packages

```bash
$ pip install -r requirements.txt
```

## Usage of the model

```python
import torch
from yacs.config import CfgNode as cdict
from lptta import normalize_model, LpTTA
from torchvision.models import vit_b_16

mu = (0.485, 0.456, 0.406)
sigma = (0.229, 0.224, 0.225)
backbone = normalize_model(vit_b_16(pretrained=True), mu, sigma)

cfg = cdict(new_allowed=True)
cfg.merge_from_file('config.yml')

model = LpTTA(model=backbone, **cfg.paras_adapt_model)
test_images = torch.rand(8, 3, 224, 224)

model(test_images)
```
