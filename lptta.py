import random
import time
from copy import deepcopy
import PIL
import torch
from torch.cuda.amp import GradScaler, autocast

import math
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from typing import Tuple, TypeVar
from torch import Tensor
from collections import OrderedDict

# This is a sample Python script.
class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore


def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
                    std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([('normalize', ImageNormalizer(mean, std)),
                          ('model', model)])
    return nn.Sequential(layers)


class LpTTA(nn.Module):
    def __init__(self, model, paras_optim, margin_e0, num_classes, reset_constant_em, arch, max_len, K, gamma, mode,
                 alpha, dataset, lambda_r=0., steps=1, episodic=False, use_label_propagation=True, use_sharpness=True):
        super().__init__()

        self.model = model
        self.margin_e0 = margin_e0 * math.log(num_classes)
        self.reset_constant_em = reset_constant_em
        self.paras_optim = paras_optim  # optimizer for SAR {lr:, momentum}
        self.episodic = episodic
        self.steps = steps
        self.use_label_propagation = use_label_propagation
        self.use_sharpness = use_sharpness

        self.configure_model()
        self.params, self.names = self.collect_params()

        self.optimizer = self.setup_optimizer()

        # state copy
        self.model_state, self.optimizer_state = deepcopy(model.state_dict()), deepcopy(self.optimizer.state_dict())

        self.arch = arch
        self.dataset = dataset
        self.feature_extractor, self.classifier = split_up_model(self.model, arch, dataset)

        self.memory_bank = GraphMemoryBank(max_len=max_len, bank=None, K=K, gamma=gamma, mode=mode, alpha=alpha)
        self.lambda_r = lambda_r
        self.original_model = deepcopy(model)


    def configure_model(self):
        """Configure model for use with SAR."""
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what SAR updates
        self.model.requires_grad_(False)
        # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def collect_params(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    def setup_optimizer(self):
        # architecture_name = self.cfg.MODEL.ARCH.lower().replace("-", "_")
        # if "vit_" in architecture_name or "swin_" in architecture_name:
        #     return SAM(self.params, torch.optim.SGD, lr=0.001, momentum=self.paras_optim['momentum'])
        # else:
        #     return SAM(self.params, torch.optim.SGD, lr=self.paras_optim['lr'], momentum=self.paras_optim['momentum'])
        if self.use_sharpness:
            return SAM(self.params, torch.optim.SGD, lr=self.paras_optim['lr'], momentum=0.9)
        else:
            return torch.optim.SGD(self.params, lr=self.paras_optim['lr'], momentum=0.9)

    def forward(self, x):
        if self.episodic:
            self.reset()

        if isinstance(x, dict):
            if 'inputs' in x:
                x = x['inputs']
            elif 'img' in x:
                x = x['img']

        self.feature_extractor.eval()
        representations = self.feature_extractor(x)
        predict = self.classifier(representations)
        predict = torch.softmax(predict, dim=1)

        predict_labels = torch.argmax(predict, dim=1)

        if self.use_label_propagation:
            refined_soft_labels, refined_labels = self.memory_bank.propagate(
                (representations.tolist(), predict.tolist()))
            refined_soft_labels = torch.tensor(refined_soft_labels, device='cuda')

            self.memory_bank.append(representations=representations, predictions=predict)

        # adaptation (include inference)
        for _ in range(self.steps):
            outputs = self.adapt(x)

        if self.use_label_propagation:
            return refined_soft_labels
        else:
            return outputs

    @torch.enable_grad()
    def adapt(self, x):

        # first step for backpropagation
        outputs = self.model(x)
        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)

        if self.use_sharpness:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # second step for backpropagation
            entropys2 = softmax_entropy(self.model(x))
            entropys2 = entropys2[filter_ids_1]
            filter_ids_2 = torch.where(entropys2 < self.margin_e0)
            loss_second = entropys2[filter_ids_2].mean(0)

            loss_second += self.l2_regularization(self.model, self.original_model, self.lambda_r)

            loss_second.backward()
            self.optimizer.second_step(zero_grad=True)

        else:
            loss += self.l2_regularization(self.model, self.original_model, self.lambda_r)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

        self.feature_extractor, self.classifier = split_up_model(self.model, self.arch, self.dataset)

        self.memory_bank.bank = [[], []]

    def l2_regularization(self,model, original_model, lambda_r):
        loss = 0
        for p, p_orig in zip(model.parameters(), original_model.parameters()):
            if p.requires_grad:
                loss += lambda_r * (p - p_orig).pow(2).sum()
        return loss

class SAM(torch.optim.Optimizer):
    # from https://github.com/davda54/sam
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def split_up_model(model, arch_name: str, dataset_name: str):
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    Input:
        model: Model to be split up
        arch_name: Name of the network
        dataset_name: Name of the dataset
    Returns:
        encoder: The encoder of the model
        classifier The classifier of the model
    """
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model,
                                                                                      model.model.pretrained_cfg[
                                                                                          "classifier"]):
        # split up models loaded from timm
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)

    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8),
                                nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1],
                                nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50",
                                                                                                        "Hendrycks2020AugMix",
                                                                                                        "Hendrycks2020Many",
                                                                                                        "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1],
                                nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute,
                                model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    # add a masking layer to the classifier
    if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        from imagenet_subsets import ImageNetXMaskingLayer, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_A_MASK

        mask = eval(f"{dataset_name.upper()}_MASK")
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))

    return encoder, classifier


class ImageNetXMaskingLayer(torch.nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """

    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]


class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

class GraphMemoryBank:
    def __init__(self, max_len, bank, K, gamma, mode, alpha, repeat=1, device='cuda', verbose=False):
        self.bank = bank if bank is not None else [[], []]
        self.K = K
        self.gamma = gamma
        self.mode = mode
        self.repeat = repeat
        self.device = device
        self.verbose = verbose
        self.max_len = max_len
        self.alpha = alpha

    def append(self, representations, predictions):
        self.bank[0].extend(representations.tolist())
        self.bank[1].extend(predictions.tolist())
        if self.max_len and len(self.bank[0]) > self.max_len:
            self.bank[0] = self.bank[0][-self.max_len:]
            self.bank[1] = self.bank[1][-self.max_len:]

    def propagate(self, buffer):
        (refined_soft_labels,
         refined_labels) = label_propagation_for_buffer_v2(buffer=buffer,
                                                                 bank=self.bank,
                                                                 K=self.K,
                                                                 gamma=self.gamma,
                                                                 mode=self.mode,
                                                                 repeat=self.repeat,
                                                                 device=self.device,
                                                                 verbose=self.verbose,
                                                                 alpha=self.alpha
                                                                 )
        return refined_soft_labels, refined_labels

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def label_propagation_for_buffer_v2(
        buffer=([], []),
        bank=([], []),
        K=50,
        gamma=1,
        mode='l2',
        repeat=1,
        device='cuda',
        verbose=True,
        alpha=0.99,
        max_iter=20
):
    import numpy as np
    import scipy
    import faiss
    from scipy.sparse.linalg import cg

    # Combine labels and features from the bank and buffer
    labels = np.array(bank[1] + buffer[1])
    features = np.array(bank[0] + buffer[0])
    # print('shape of labels:', labels.shape)
    num_classes = labels.shape[-1]
    # Validate inputs
    if None in labels or None in features:
        raise ValueError("Some samples do not have pseudo labels or features")
    if len(labels) != len(features):
        raise ValueError(
            "Length of labels and features must be the same, but got different lengths. " + f' length of labels: {len(labels)}, length of features: {len(features)}')

    # kNN search for the graph
    X = features.astype('float32')
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()

    index = faiss.GpuIndexFlatIP(res, X.shape[1], flat_config)

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    Nidx = index.ntotal

    c = time.time()
    K = min(K, N - 1)
    D, I = index.search(X, K + 1)
    elapsed = time.time() - c
    # print('kNN Search done in %d seconds' % elapsed)

    # create graph
    D = D[:, 1:]
    I = I[:, 1:]

    # normalize D
    D_min = np.min(D, axis=1)[:, np.newaxis]
    D_max = np.max(D, axis=1)[:, np.newaxis]
    D = (D - D_min) / (D_max - D_min + 1e-8)
    D = D**gamma

    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (K, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))

    # Set the diagonal of W to 1
    W.setdiag(1)

    W = W + W.T

    # Normalize the affinity matrix
    # W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))

    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D @ W @ D

    # Initiliaze the y vector for each class
    Z = np.zeros((N, num_classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn

    top_k = 1
    # obtain top k labels and set them as potential labels
    buffer_labels = np.array(buffer[1])
    top_k_indices = np.argsort(buffer_labels, axis=-1)[..., -top_k:]
    potential_labels = set(top_k_indices.flatten())
    # to list
    potential_labels = list(potential_labels)

    # Z = np.zeros_like(labels)
    # tol = 1e-6
    #
    # with Pool() as pool:
    #     results = pool.map(process_label, [(A, labels[:, i], tol, max_iter) for i in potential_labels])
    #
    # for i, result in enumerate(results):
    #     Z[:, potential_labels[i]] = result

    for i in potential_labels:
        y = labels[:, i]
        y = y / (y.sum() + 1e-8)

        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    labels = np.argmax(Z, axis=-1)

    # Return the labels corresponding to the buffer
    refined_labels = labels[len(bank[1]):]
    refined_soft_labels = Z[len(bank[1]):]
    return refined_soft_labels, refined_labels
