import collections
import CLIP_.clip as clip
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

from models.decomp import maskLoss
from models.histogram import HistLoss
from models.structure import SuperPixel
from models.pyramid import MSEPyramidLoss
from models.edge import GradLoss


def compute_sine_theta(s1, s2):  # s1 and s2 aret two segments to be uswed
    # s1, s2 (2, 2)
    v1 = s1[1, :] - s1[0, :]
    v2 = s2[1, :] - s2[0, :]
    # print(v1, v2)
    sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2))
    return sine_theta


def get_sdf(phi, method='skfmm', **kwargs):
    if method == 'skfmm':
        import skfmm
        phi = (phi - 0.5) * 2
        if (phi.max() <= 0) or (phi.min() >= 0):
            return np.zeros(phi.shape).astype(np.float32)
        sdf = []
        for img in phi:
            sd = skfmm.distance(img, dx=1)

            flip_negative = kwargs.get('flip_negative', True)
            if flip_negative:
                sd = np.abs(sd)

            truncate = kwargs.get('truncate', 10)
            sd = np.clip(sd, -truncate, truncate)
            # print(f"max sd value is: {sd.max()}")

            zero2max = kwargs.get('zero2max', True)
            if zero2max and flip_negative:
                sd = sd.max() - sd
            elif zero2max:
                raise ValueError

            normalize = kwargs.get('normalize', 'sum')
            if normalize == 'sum':
                sd /= sd.sum()
            elif normalize == 'to1':
                sd /= sd.max()
            sdf.append(sd[None])
        return torch.FloatTensor(np.clip(np.concatenate(sdf, axis=0), 0, 1))


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.percep_loss = args.percep_loss

        self.train_with_clip = args.train_with_clip
        self.clip_weight = args.clip_weight
        self.start_clip = args.start_clip

        self.clip_conv_loss = args.clip_conv_loss
        self.xing_loss_weight = args.xing_loss_weight
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.clip_text_guide = args.clip_text_guide

        self.losses_to_apply = self.get_losses_to_apply()

        self.loss_mapper = \
            {
                "clip": CLIPLoss(args),
                "clip_conv_loss": CLIPConvLoss(args),
                "L2": MSEPyramidLoss(device=args.device),
                "Hist": HistLoss(device=args.device),
                "SDF": torch.nn.MSELoss(reduction='none'),
                "Xing_Loss": self.Xing_Loss,
                "Edge": GradLoss().to(args.device),
                "Mask": maskLoss
            }
        self.structure = SuperPixel(args.device, mode='sscolor')

    def distance(self, shapes):
        loss = 0
        for path1 in shapes:
            x1 = path1.points
            for path2 in shapes:
                x2 = path2.points
                loss += ((x1 - x2) ** 2).mean()
        return loss

    def Xing_Loss(self, shapes, scale=1):  # x[ npoints,2]
        loss = 0.
        # print(len(x_list))
        for shape in shapes:
            x = shape.points
            seg_loss = 0.
            N = x.size()[0]
            x = torch.cat([x, x[0, :].unsqueeze(0)], dim=0)  # (N+1,2)
            segments = torch.cat([x[:-1, :].unsqueeze(1), x[1:, :].unsqueeze(1)], dim=1)  # (N, start/end, 2)
            assert N % 3 == 0, 'The segment number is not correct!'
            segment_num = int(N / 3)
            for i in range(segment_num):
                cs1 = segments[i * 3, :, :]  # start control segs
                cs2 = segments[i * 3 + 1, :, :]  # middle control segs
                cs3 = segments[i * 3 + 2, :, :]  # end control segs
                # print('the direction of the vectors:')
                # print(compute_sine_theta(cs1, cs2))
                direct = (compute_sine_theta(cs1, cs2) >= 0).float()
                opst = 1 - direct  # another direction
                sina = compute_sine_theta(cs1, cs3)  # the angle between cs1 and cs3
                seg_loss += direct * torch.relu(- sina) + opst * torch.relu(sina)
                # print(direct, opst, sina)
            seg_loss /= segment_num

            templ = seg_loss
            loss += templ * scale  # area_loss * scale

        return loss / (len(shapes))

    def get_losses_to_apply(self):
        losses_to_apply = []
        if self.percep_loss != "none":
            losses_to_apply.append(self.percep_loss)
        if self.train_with_clip and self.start_clip == 0:
            losses_to_apply.append("clip")
        if self.clip_conv_loss:
            losses_to_apply.append("clip_conv_loss")
        if self.clip_text_guide:
            losses_to_apply.append("clip_text")
        losses_to_apply.append("L2")

        # losses_to_apply.append("Mask")
        # losses_to_apply.append("SDF")
        # losses_to_apply.append("Sizes")
        # losses_to_apply.append("Hist")
        # losses_to_apply.append("Edge")
        # losses_to_apply.append("Distance")
        # losses_to_apply.append("Xing_Loss")
        return losses_to_apply

    def update_losses_to_apply(self, epoch):
        if "clip" not in self.losses_to_apply:
            if self.train_with_clip:
                if epoch > self.start_clip:
                    self.losses_to_apply.append("clip")

    def forward(self, svg_img, targets, masks, epoch, stacked=None, mode="train", points=None, sizes=None,
                im_forsdf=None):
        loss = 0
        self.update_losses_to_apply(epoch)

        losses_dict = dict.fromkeys(
            self.losses_to_apply, torch.tensor([0.0]).to(self.args.device))
        loss_coeffs = dict.fromkeys(self.losses_to_apply, 1.0)
        loss_coeffs["clip"] = self.clip_weight
        loss_coeffs["clip_text"] = self.clip_text_guide
        loss_coeffs["Xing_Loss"] = self.xing_loss_weight
        loss_coeffs["Mask"] = 1
        loss_coeffs["SDF"] = 100
        loss_coeffs["Sizes"] = 1e-1
        loss_coeffs["Edge"] = 10

        for loss_name in self.losses_to_apply:
            if loss_name in ["clip_conv_loss"]:
                conv_loss = self.loss_mapper[loss_name](
                    svg_img, targets, mode)
                for layer in conv_loss.keys():
                    losses_dict[layer] = conv_loss[layer]
            elif loss_name == "L2":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    svg_img, targets).mean()
            elif loss_name == "Xing_Loss":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    points, scale=1)
            elif loss_name == "Sizes":
                losses_dict[loss_name] = sizes.sum()
            elif loss_name == "SDF":
                sdf = (im_forsdf[:, 3]).detach().cpu().numpy()
                losses_dict[loss_name] = (
                            get_sdf(sdf, normalize='to1').to(self.args.device) * self.loss_mapper[loss_name](
                        svg_img, targets).sum(1)).mean()
            elif loss_name == "Mask":
                losses_dict[loss_name] = self.loss_mapper[loss_name](stacked, targets, masks)
            else:
                losses_dict[loss_name] = (self.loss_mapper[loss_name](
                    svg_img, targets, mode)).mean()

        for key in self.losses_to_apply:
            if key != "L2":
                losses_dict[key] = losses_dict[key] * loss_coeffs[key]
        # print(losses_dict)
        return losses_dict


class CLIPLoss(torch.nn.Module):
    def __init__(self, args, text_prompt=None):
        super(CLIPLoss, self).__init__()

        self.args = args
        self.model, clip_preprocess = clip.load(
            'ViT-B/32', args.device, jit=False)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [clip_preprocess.transforms[0], clip_preprocess.transforms[-1]])  # clip normalisation
        # self.preprocess = transforms.Compose([clip_preprocess.transforms[-1]])  # clip normalisation
        self.device = args.device
        self.NUM_AUGS = args.num_aug_clip
        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.calc_target = True
        self.include_target_in_aug = args.include_target_in_aug
        self.counter = 0
        self.augment_both = args.augment_both
        self.text_prompt = text_prompt

    def forward(self, sketches, targets, mode="train"):
        if self.calc_target:
            targets_ = self.preprocess(targets).to(self.device)
            self.targets_features = self.model.encode_image(targets_).detach()
            self.calc_target = False
        if self.text_prompt is not None:
            text_input = clip.tokenize([self.text_prompt]).to(self.device)
            text_features = self.model.encode_text(text_input).detach()

        if mode == "eval":
            # for regular clip distance, no augmentations
            with torch.no_grad():
                sketches = self.preprocess(sketches).to(self.device)
                sketches_features = self.model.encode_image(sketches)
                return 1. - torch.cosine_similarity(sketches_features, self.targets_features)

        loss_clip = 0
        sketch_augs = []
        for n in range(self.NUM_AUGS):
            augmented_pair = self.augment_trans(torch.cat([sketches, targets]))
            sketch_augs.append(augmented_pair[0].unsqueeze(0))

        sketch_batch = torch.cat(sketch_augs)
        sketch_features = self.model.encode_image(sketch_batch)

        for n in range(self.NUM_AUGS):
            loss_clip += (1. - torch.cosine_similarity(
                sketch_features[n:n + 1], self.targets_features, dim=1))
        if self.text_prompt is not None:
            for n in range(self.NUM_AUGS):
                loss_clip += (1. - torch.cosine_similarity(
                    sketch_features[n:n + 1], text_features, dim=1))
        self.counter += 1
        return loss_clip


class LPIPS(torch.nn.Module):
    def __init__(self, pretrained=True, normalize=True, pre_relu=True, device=None):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(LPIPS, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        self.normalize = normalize
        self.pretrained = pretrained
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        self.augment_trans = transforms.Compose(augemntations)
        self.feature_extractor = LPIPS._FeatureExtractor(
            pretrained, pre_relu).to(device)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0)
        ys = torch.cat(img_augs, dim=0)

        pred = self.feature_extractor(xs)
        target = self.feature_extractor(ys)

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]

        # TODO(mgharbi) Apply Richard's linear weights?

        if self.normalize:
            diffs = [torch.sum((p - t) ** 2, 1)
                     for (p, t) in zip(pred, target)]
        else:
            # mean instead of sum to avoid super high range
            diffs = [torch.mean((p - t) ** 2, 1)
                     for (p, t) in zip(pred, target)]

        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs)

    class _FeatureExtractor(torch.nn.Module):
        def __init__(self, pretrained, pre_relu):
            super(LPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg16(pretrained=pretrained).features

            self.breakpoints = [0, 4, 9, 16, 23, 30]
            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i + 1] -= 1

            # Split at the maxpools
            for i, b in enumerate(self.breakpoints[:-1]):
                ops = torch.nn.Sequential()
                for idx in range(b, self.breakpoints[i + 1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                # print(ops)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer("shift", torch.Tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.Tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints) - 1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)
            return feats


class L2_(torch.nn.Module):
    def __init__(self):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(L2_, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        # LOG.warning("LPIPS is untested")

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        pred = torch.cat(sketch_augs, dim=0)
        target = torch.cat(img_augs, dim=0)
        diffs = [torch.square(p - t).mean() for (p, t) in zip(pred, target)]
        return sum(diffs)


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model, device, mask_cls="none", apply_mask=False, mask_attention=False):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None
        self.device = device
        self.n_channels = 3
        self.kernel_h = 32
        self.kernel_w = 32
        self.step = 32
        self.num_patches = 49
        self.mask_cls = mask_cls
        self.apply_mask = apply_mask
        self.mask_attention = mask_attention

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x, masks=None, mode="train"):
        masks_flat = torch.ones((x.shape[0], 50, 768)).to(self.device)  # without any effect
        attn_map = None
        if masks is not None and self.apply_mask:
            x_copy = x.detach().clone()

            patches_x = x_copy.unfold(2, self.kernel_h, self.step).unfold(3, self.kernel_w, self.step).reshape(-1,
                                                                                                               self.n_channels,
                                                                                                               self.num_patches,
                                                                                                               32, 32)
            # split the masks into patches (the same input patches to the transformer)
            # shape is (batch_size, channel, num_patches, patch_size, patch_size) = (5, 3, 49, 32, 32)
            patches_mask = masks.unfold(2, self.kernel_h, self.step).unfold(3, self.kernel_w, self.step).reshape(-1,
                                                                                                                 self.n_channels,
                                                                                                                 self.num_patches,
                                                                                                                 32, 32)
            # masks_ is a binary mask (batch_size, 1, 7, ,7) to say which patch should be masked out
            masks_ = torch.ones((x.shape[0], 1, 7, 7)).to(self.device)
            for i in range(masks.shape[0]):
                for j in range(self.num_patches):
                    # we mask a patch if more than 20% of the patch is masked
                    zeros = (patches_mask[i, 0, j] == 0).sum() / (self.kernel_w * self.kernel_h)
                    if zeros > 0.2:
                        masks_[i, :, j // 7, j % 7] = 0

            if self.mask_attention:
                mask2 = masks_[:, 0].reshape(-1, 49).to(self.device)  # .to(device) shape (5, 49)
                mask2 = torch.cat([torch.ones(mask2.shape[0], 1).to(self.device), mask2], dim=-1)
                mask2 = mask2.unsqueeze(1)
                attn_map = mask2.repeat(1, 50, 1).to(self.device)  # 5, 50, 50
                # attn_map = torch.bmm(mask2.permute(0,2,1), mask2) # 5, 50, 50
                attn_map[:, 0, 0] = 1
                # attn_map[:,:,0] = 1
                attn_map = 1 - attn_map
                indixes = (attn_map == 0).nonzero()  # shape [136, 2] [[aug_im],[index]]
                attn_map = attn_map.repeat(12, 1, 1).bool()  # [60, 50, 50]
                # attn_map = attn_map.repeat(12,1,1).bool()

            # masks_ = torch.nn.functional.interpolate(masks, size=7, mode='nearest')
            # masks_[masks_ < 0.5] = 0
            # masks_[masks_ >=0.5] = 1

            # masks_flat's shape is (5, 49), for each image in the batch we have 49 flags indicating if to mask the i'th patch or not
            masks_flat = masks_[:, 0].reshape(-1, self.num_patches)
            # indixes = (masks_flat == 0).nonzero() # shape [136, 2] [[aug_im],[index]]
            # for t in indixes:
            #     b_num, y, x_ = t[0], t[1] // 7, t[1] % 7
            #     x_copy[b_num, :, 32 * y: 32 * y + 32, 32 * x_: 32 * x_ + 32] = 0
            # now we add the cls token mask, it's all ones for now since we want to leave it
            # now the shape is (5, 50) where the first number in each of the 5 rows is 1 (meaning - son't mask the cls token)
            masks_flat = torch.cat([torch.ones(masks_flat.shape[0], 1).to(self.device), masks_flat],
                                   dim=1)  # include cls by default
            # now we duplicate this from (5, 50) to (5, 50, 768) to match the tokens dimentions
            masks_flat = masks_flat.unsqueeze(2).repeat(1, 1, 768)  # shape is (5, 50, 768)

            # masks_flat = masks_[:,0].reshape(-1, 49)#.to(device) shape (5, 49)


        elif self.mask_cls != "none":
            if self.mask_cls == "only_cls":
                masks_flat = torch.zeros((5, 50, 768)).to(self.device)
                masks_flat[:, 0, :] = 1
            elif self.mask_cls == "cls_out":
                masks_flat[:, 0, :] = 0

        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        # fc_features = self.clip_model.encode_image(x, attn_map, mode).float()
        # Each featuremap is in shape (5,50,768) - 5 is the batchsize(augment), 50 is cls + 49 patches, 768 is the dimension of the features
        # for each k (each of the 12 layers) we only take the vectors
        # if masks is not None and self.apply_mask:
        featuremaps = [self.featuremaps[k] * masks_flat for k in range(12)]
        # featuremaps = [self.featuremaps[k][masks_flat == 1] for k in range(12)]

        # else:
        # featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, args, clip_model_name="ViT-B/32"):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = clip_model_name
        #                               0  1  2     3  4  5  6  7     8  9  10   11   0
        self.clip_conv_layer_weights = [0, 0, 0.35, 0, 0, 0, 0, 0.45, 0, 0, 0.5, 0.9, 0]
        assert self.clip_model_name in [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_conv_loss_type = args.clip_conv_loss_type
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, args.device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model, args.device)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

        self.args = args

        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = args.device
        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        #         y = self.target_transform(target).to(self.args.device)
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())

        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)

        for layer, w in enumerate(self.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w

        if self.clip_fc_loss_weight:
            # fc distance is always cos
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                                                   ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]
