"""
FusionToVLM:
in : health (B, d_fusion) float32   the fusion stack's health vector
     prompt text                    the turbine brief, tokenised normally
out: logits / generated text        the VLM reading both at once

WHAT THIS IS

The bridge between MultiModalFusion and a frozen Qwen3-VL. The fusion vector is
projected into the language model's embedding space and spliced into the token
stream as SOFT TOKENS -- vectors that occupy token positions but were never in
the vocabulary. This is the LLaVA pattern: a vision encoder's output is
projected to the LM's width and prepended, and only the projector trains.

WHY A VLM AND NOT A TEXT-ONLY LM

Not for the vision tower. A VLM's language model has already been trained with
non-text embeddings flowing through its residual stream, so it is pre-adapted to
exactly the regime this puts it in: some input positions are not words. A pure
text LM has never seen that and has to learn it from the projector alone.

THE VISION TOWER IS STILL NOT REDUNDANT WITH rgb_ViT

They feed different consumers. rgb_ViT goes into the fusion, so blade imagery
reaches the detection heads and participates in presence masking like any other
modality. The VLM's own tower is for when the report needs the model to LOOK at
a specific photograph rather than read a compressed summary of one. Running both
on the same image is duplication only if you wanted the same thing twice.

THE PROJECTOR IS UNTRAINED AND THEREFORE MEANINGLESS

Freshly initialised, these soft tokens are noise -- a 235B model reads noise
exactly as badly as a 2B one. They mean something only after training on paired
(sensor window, text) examples. scripts/turbine_report.py generates the text
side from residuals, status logs and templates; data_io/kelmarsh_io.py generates
the sensor side. Manufacturing those pairs is the real work; the checkpoint is a
config line.

    from models.vlm_bridge import FusionToVLM
    bridge = FusionToVLM("Qwen/Qwen3-VL-8B-Instruct", d_fusion=128)
    out = bridge(health=health, text=[brief], labels_text=[assessment])
    out.loss.backward()          # only the projector has gradients
"""

import torch
import torch.nn as nn

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError as e:                                   # pragma: no cover
    raise ImportError(
        "needs a transformers with Qwen3-VL support: "
        "pip install -U 'transformers>=5.0'") from e

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class FusionProjector(nn.Module):
    """(B, d_fusion) -> (B, n_soft, d_lm), scale-matched to token embeddings.

    Two things here are load-bearing and easy to get wrong.

    n_soft > 1 because one token is a bottleneck: attention can only weight a
    position, not decompose it, so a single vector forces the LM to unpack the
    whole health state from one attention target. Eight gives it somewhere to
    put separable facts.

    The output is RESCALED to the norm of the model's real token embeddings.
    A freshly initialised Linear emits vectors whose norm is unrelated to the
    embedding distribution -- typically far too small, in which case attention
    ignores them and the projector gets almost no gradient, or far too large,
    in which case they swamp the prompt and generation degenerates. Matching
    the scale puts the soft tokens in the same regime the LM's layers expect.
    """

    def __init__(self, d_fusion, d_lm, n_soft=8, target_norm=1.0):
        super().__init__()
        self.n_soft, self.d_lm = n_soft, d_lm
        # A SHARED trunk, then per-slot FiLM. The obvious alternative --
        # Linear(4*d_lm, n_soft*d_lm) to emit all K tokens at once -- costs
        # 134M parameters at d_lm 2048 and 537M at 4096, which is not an
        # adapter but a small model, and it would memorise the few thousand
        # pairs this is trainable on. Broadcasting one trunk output and giving
        # each slot its own scale and shift keeps the K tokens distinct for
        # 2 * n_soft * d_lm extra weights -- 32k here instead of 134M.
        self.net = nn.Sequential(
            nn.LayerNorm(d_fusion),
            nn.Linear(d_fusion, d_lm),
            nn.GELU(),
            nn.Linear(d_lm, d_lm),
        )
        self.slot_scale = nn.Parameter(torch.ones(n_soft, d_lm))
        self.slot_shift = nn.Parameter(torch.randn(n_soft, d_lm) * 0.02)
        self.out_norm = nn.LayerNorm(d_lm)
        # buffer, not a constant: it is measured from the loaded checkpoint's
        # embedding table and must follow the model onto GPU and into a save
        self.register_buffer("target_norm", torch.tensor(float(target_norm)))

    def forward(self, health):
        h = self.net(health).unsqueeze(1)                    # (B, 1, d_lm)
        z = self.out_norm(h * self.slot_scale + self.slot_shift)  # (B, K, d_lm)
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.target_norm
        return z


class FusionToVLM(nn.Module):
    """Frozen Qwen3-VL + a trainable projector from the fusion vector."""

    def __init__(self, model_id=DEFAULT_MODEL, d_fusion=128, n_soft=8,
                 freeze_vlm=True, dtype="auto", device_map=None,
                 attn_implementation=None):
        super().__init__()
        kw = dict(dtype=dtype)
        if device_map is not None:
            kw["device_map"] = device_map
        if attn_implementation is not None:
            kw["attn_implementation"] = attn_implementation
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kw)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        # Qwen3-VL nests the language model's config under text_config; reading
        # hidden_size off the top-level config gets the wrong number on some
        # checkpoints, and a wrong projector width fails at concat time with a
        # shape error that names neither the model nor the projector.
        cfg = self.vlm.config
        d_lm = getattr(getattr(cfg, "text_config", cfg), "hidden_size", None)
        if d_lm is None:
            raise RuntimeError(f"could not read hidden_size from {type(cfg)}")

        emb = self.vlm.get_input_embeddings().weight
        target = emb.detach().float().norm(dim=-1).median().item()
        # The projector stays in FLOAT32 while the frozen VLM is whatever dtype
        # it loaded in (usually bfloat16). Casting the projector down to match
        # breaks on the first forward -- LayerNorm rejects a bf16 parameter fed
        # a float32 input -- and it is the wrong fix anyway: this is the only
        # module that learns, and fp32 master weights for the trainable part
        # with a low-precision frozen backbone is the standard mixed-precision
        # arrangement. The cast happens on the OUTPUT, at the splice.
        self.projector = FusionProjector(d_fusion, d_lm, n_soft, target)
        self.lm_dtype = emb.dtype

        self.frozen = freeze_vlm
        if freeze_vlm:
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False

        self.d_lm, self.n_soft = d_lm, n_soft

    def train(self, mode=True):
        """Pin the VLM to eval even when the parent trains.

        Same reason RGBEncoder does it: a frozen backbone left in train mode
        runs its dropout, so the "frozen" features differ between forward
        passes and the projector is fitting noise.
        """
        super().train(mode)
        if self.frozen:
            self.vlm.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # -- building the spliced sequence ---------------------------------------
    def build_inputs(self, health, text, labels_text=None, max_length=2048):
        """health + prompt (+ target) -> inputs_embeds, attention_mask, labels.

        Layout, left to right:

            [ soft_0 .. soft_{K-1} ] [ prompt tokens ] [ target tokens ]
              projector output         the brief         the assessment
              labels = -100            labels = -100     labels = token ids

        The two -100 spans matter. Loss is only defined on the target: the
        prompt is given, and the soft tokens have no token id to predict at
        all, so scoring either one trains the model to reproduce its own input.
        """
        dev = next(self.vlm.parameters()).device
        # fp32 through the projector, then down to the LM's dtype for the splice
        soft = self.projector(health.to(dev, torch.float32))      # (B, K, d_lm)
        soft = soft.to(self.lm_dtype)
        B = soft.shape[0]

        embed = self.vlm.get_input_embeddings()
        seqs, labs = [], []
        for i in range(B):
            p = self.tokenizer(text[i], return_tensors="pt",
                               truncation=True, max_length=max_length,
                               add_special_tokens=True).input_ids[0].to(dev)
            ids = p
            lab = torch.full_like(p, -100)
            if labels_text is not None:
                t = self.tokenizer(labels_text[i], return_tensors="pt",
                                   truncation=True, max_length=max_length,
                                   add_special_tokens=False).input_ids[0].to(dev)
                ids = torch.cat([p, t])
                lab = torch.cat([lab, t])                # supervise target only
            seqs.append(ids)
            labs.append(lab)

        # right-pad to the longest in the batch; pad positions are masked out of
        # attention AND set to -100 so they never contribute to the loss
        n = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids = torch.full((B, n), pad, dtype=torch.long, device=dev)
        lab = torch.full((B, n), -100, dtype=torch.long, device=dev)
        att = torch.zeros((B, n), dtype=torch.long, device=dev)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            ids[i, :len(s)], lab[i, :len(l)], att[i, :len(s)] = s, l, 1

        txt = embed(ids).to(self.lm_dtype)
        inputs_embeds = torch.cat([soft, txt], dim=1)
        attention_mask = torch.cat(
            [torch.ones(B, self.n_soft, dtype=torch.long, device=dev), att], 1)
        labels = torch.cat(
            [torch.full((B, self.n_soft), -100, dtype=torch.long, device=dev),
             lab], dim=1)
        return inputs_embeds, attention_mask, labels

    def forward(self, health, text, labels_text=None, **kw):
        emb, att, lab = self.build_inputs(health, text, labels_text, **kw)
        return self.vlm(inputs_embeds=emb, attention_mask=att,
                        labels=lab if labels_text is not None else None)

    @torch.no_grad()
    def generate(self, health, text, max_new_tokens=512, **kw):
        emb, att, _ = self.build_inputs(health, text, labels_text=None)
        out = self.vlm.generate(inputs_embeds=emb, attention_mask=att,
                                max_new_tokens=max_new_tokens, **kw)
        # with inputs_embeds there are no input ids to strip -- generate()
        # returns only the new tokens, unlike the input_ids path
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


if __name__ == "__main__":
    # A demo, not a test. The splice checks (scale matching, label masking,
    # gradient routing) now live in tests/vlm/smoke_test.py.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    a = ap.parse_args()

    torch.manual_seed(0)
    print(f"loading {a.model} ...")
    bridge = FusionToVLM(a.model, d_fusion=128, n_soft=8)
    n_tr = sum(p.numel() for p in bridge.trainable_parameters())
    n_fz = sum(p.numel() for p in bridge.parameters()) - n_tr

    emb, att, lab = bridge.build_inputs(
        torch.randn(2, 128),
        ["Turbine 3: bearing residual +3.2 C, rising 11 days.",
         "Turbine 1: residuals within normal scatter."],
        ["Investigate the generator cooling circuit.", "No action required."])

    print(f"d_lm            {bridge.d_lm}")
    print(f"soft tokens     {bridge.n_soft}")
    print(f"trainable       {n_tr:,}   (projector only)")
    print(f"frozen          {n_fz:,}")
    print(f"embedding norm  {bridge.projector.target_norm.item():.3f}")
    print(f"inputs_embeds   {tuple(emb.shape)}  = {bridge.n_soft} soft + "
          f"{emb.shape[1] - bridge.n_soft} text")
    print(f"supervised      {int((lab != -100).sum())} of {lab.numel()} positions")
