"""
RGBEncoder:
in : x    (B, 3, H, W) float32   ImageNet-normalised blade/nacelle imagery
     mask (B,) float32 or None   1 = usable frame, 0 = camera dead this window
out:      (B, 1 + P, d) float32  one global token then P patch tokens

B = Batch     - how many frames you process at once
P = Patches   - (H/14) * (W/14) for a patch-14 backbone; 1369 at 518x518
d = token width (128, matching every other encoder)

WHY A FROZEN BACKBONE

Blade imagery is the scarcest modality here -- a handful of inspection flights,
not years of 10-minute rows -- and an 86M-parameter ViT finetuned on that would
memorise it outright. DINOv2 features are already strong for dense
correspondence and defect-like texture without any turbine-specific training,
so the trainable surface is deliberately tiny: a LayerNorm and one Linear,
about 100k parameters against 86.6M frozen.

That ratio is the design. If blade labels ever become plentiful, unfreezing the
last few blocks is the natural next step, and `freeze_backbone=False` exists for
it -- but it should be a deliberate decision, not a default.

    from models.rgb_ViT import RGBEncoder
    from models.common import PerceiverResampler
    enc = RGBEncoder(d_model=128)
    tokens  = enc(imgs, mask)                       # (B, 1+P, 128)
    latents = PerceiverResampler(d_model=128)(tokens)   # (B, 32, 128)
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:                              # pragma: no cover
    raise ImportError("RGBEncoder needs timm: pip install timm") from e


class RGBEncoder(nn.Module):
    """Frozen DINOv2 ViT -> d_model tokens the fusion can consume."""

    def __init__(self, d_model=128, model_name="vit_base_patch14_dinov2.lvd142m",
                 pretrained=True, freeze_backbone=True, keep_cls=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained,
                                     num_classes=0)
        self.keep_cls = keep_cls
        self.frozen = freeze_backbone
        if freeze_backbone:
            self.vit.eval()
            for p in self.vit.parameters():
                p.requires_grad = False

        self.patch = self.vit.patch_embed.patch_size[0]
        self.proj = nn.Sequential(nn.LayerNorm(self.vit.embed_dim),
                                  nn.Linear(self.vit.embed_dim, d_model))

        # DINOv2 was trained on ImageNet-normalised input and its features
        # degrade badly on raw [0,1] or unnormalised pixels. Buffers, not
        # constants, so `preprocess` follows the model onto MPS/CUDA.
        self.register_buffer("pix_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pix_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # -- keep the backbone in eval even when the parent goes to train ---------
    def train(self, mode=True):
        """Pin a frozen backbone to eval().

        Without this, `model.train()` re-enables the ViT's dropout and
        stochastic depth, so the same image yields different "frozen" features
        on every step -- noise the downstream head cannot learn through, and
        invisible unless you diff two forward passes.
        """
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def preprocess(self, x):
        """[0,1] pixels -> ImageNet-normalised. Skip if your loader does it."""
        return (x - self.pix_mean) / self.pix_std

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        assert C == 3, f"expected 3 colour channels, got {C}"
        # a patch-14 backbone silently mis-tiles a non-multiple size rather than
        # raising, and the resulting token grid does not correspond to the image
        assert H % self.patch == 0 and W % self.patch == 0, (
            f"{H}x{W} is not a multiple of patch size {self.patch}; "
            f"resize to e.g. {H // self.patch * self.patch}x"
            f"{W // self.patch * self.patch}")

        if self.frozen:
            with torch.no_grad():
                feats = self.vit.forward_features(x)
        else:
            feats = self.vit.forward_features(x)

        n_prefix = self.vit.num_prefix_tokens
        if self.keep_cls and n_prefix > 0:
            # CLS first, then patches. DINOv2's CLS is its strongest global
            # descriptor -- "is this blade damaged" -- while the patches carry
            # "where". Dropping it throws away the global summary and leaves the
            # resampler to rebuild it from 1369 local views. Registers (prefix
            # tokens beyond CLS) ARE dropped: they are scratch space for the
            # backbone's own attention and carry no image content.
            tokens = torch.cat([feats[:, :1], feats[:, n_prefix:]], dim=1)
        else:
            tokens = feats[:, n_prefix:]

        tokens = self.proj(tokens)

        # A dead camera is inter-modality absence, which MultiModalFusion's
        # `present` already masks out of every attention. This zeroes the block
        # anyway so a bad frame cannot leak through a caller that forgot to set
        # `present` -- cheap, and it fails safe rather than silently.
        if mask is not None:
            tokens = tokens * mask.view(B, 1, 1).to(tokens.dtype)
        return tokens


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = RGBEncoder(d_model=128, pretrained=False).eval()
    x = torch.rand(2, 3, 518, 518)
    x = enc.preprocess(x)

    with torch.no_grad():
        out = enc(x)
    n_tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_fz = sum(p.numel() for p in enc.parameters() if not p.requires_grad)
    print(f"tokens            {tuple(out.shape)}  = 1 CLS + "
          f"{(518 // enc.patch) ** 2} patches")
    print(f"trainable         {n_tr:,}")
    print(f"frozen            {n_fz:,}")

    enc.train()
    print(f"vit stays eval    {not enc.vit.training}")

    # the frozen path must be deterministic in TRAIN mode too, which is the
    # whole point of the train() override
    with torch.no_grad():
        a, b = enc(x), enc(x)
    print(f"deterministic     {torch.equal(a, b)}")

    out = enc(x)
    out.sum().backward()
    print(f"proj gets grad    {enc.proj[1].weight.grad is not None}")
    print(f"vit gets no grad  {enc.vit.patch_embed.proj.weight.grad is None}")

    with torch.no_grad():
        dead = enc(x, mask=torch.tensor([1.0, 0.0]))
    print(f"mask zeroes frame {bool((dead[1] == 0).all())} "
          f"and keeps the other {bool((dead[0] != 0).any())}")

    try:
        enc(torch.rand(1, 3, 500, 500))
    except AssertionError as e:
        print(f"rejects bad size  {str(e)[:52]}...")
