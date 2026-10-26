"""
Soft prompt tuning, CoOp-style (Zhou et al. 2022). Third of the three
proposed-fix methods, few-shot (same K/seed grid as cache_adapter.py, for a
fair comparison): instead of the project's 5 hand-written text templates
("a 3D model of a {}", etc.), learn M=16 continuous context vectors via
backprop through the frozen text encoder on the same K-shot support set.
The point encoder stays completely frozen throughout -- only the context
vectors are learned.

Three text-tower families, verified by reading each model's actual source
this session (not assumed):
  - open_clip family (ulip2, openshape, uni3d): all three call
    open_clip.create_model_and_transforms(...) directly. Standard attributes
    token_embedding / positional_embedding / transformer / ln_final /
    text_projection.
  - OpenAI-CLIP-style family (jm3d via ULIP_PointBERT's base encode_text,
    pointclipv2 via PointCLIP_V2/zeroshot_cls/clip/model.py's CLIP class):
    attribute-for-attribute identical to the open_clip family (confirmed by
    reading both source files), same wrapper works for both groups.
  - ImageBind (pointbind): structurally different -- modality_preprocessors
    / modality_trunks / modality_heads / modality_postprocessors dicts keyed
    by ModalityType.TEXT, not a single encode_text call. Its TextPreprocessor
    still has its own token_embedding + pos_embed (confirmed by reading
    imagebind/multimodal_preprocessors.py), so the same splice-the-context-
    vectors-into-token-embeddings recipe applies, just through this
    different call path.

Standard CoOp prompt construction (matching the official CoOp codebase):
tokenize "{M placeholder tokens} {classname}.", embed it once via the
model's own token_embedding, then REPLACE the M placeholder positions'
embeddings (position 1..M, right after the start-of-text token) with the
learnable context vectors, leaving the start token, classname tokens, and
end-of-text token's embeddings untouched. The end-of-text position for
final pooling is found via the ORIGINAL (unmodified) token ids' argmax,
exactly as each model's own encode_text already does -- splicing only
changes embeddings, never the token id tensor.

Usage:
  python soft_prompt_tuning.py --model ulip2
  python soft_prompt_tuning.py --model all
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import task_definitions as td
from run_robustness_condition import MODEL_LIB, FEATURE_CACHE_ROOT
from cache_adapter import K_SHOTS, SEEDS, stratified_kshot_indices

N_CTX = 16
N_EPOCHS = 60
LR = 2e-3
OUT_DIR = Path('/msc/home/q45emuni/miccai_2026/shapemi/scripts/soft_prompt_results')

OPEN_CLIP_FAMILY = {'ulip2', 'openshape', 'uni3d'}
OPENAI_CLIP_FAMILY = {'jm3d', 'pointclipv2'}
IMAGEBIND_FAMILY = {'pointbind'}


# ---------------------------------------------------------------------------
# Family 1+2: open_clip / OpenAI-CLIP-style (identical attribute pattern)
# ---------------------------------------------------------------------------

class CLIPStylePromptLearner(nn.Module):
    """Learns M shared context vectors, spliced into each class's tokenized
    prompt embedding right after the start token. Works for both the
    open_clip family and the OpenAI-CLIP-style family (jm3d, pointclipv2)
    since their text towers are attribute-identical."""

    def __init__(self, clip_text_module, tokenizer, class_names, n_ctx=N_CTX, device='cuda',
                 needs_permute=False):
        super().__init__()
        self.clip = clip_text_module
        # Two sub-families share this attribute pattern but differ in
        # calling convention: modern open_clip's encode_text operates
        # batch-first throughout (no permute); the older vendored clip.py
        # style (jm3d's ULIP_WITH_IMAGE base class, PointCLIP V2's own
        # clip/model.py) permutes to seq-first (LND) around the transformer
        # call, matching the classic OpenAI CLIP implementation.
        self.needs_permute = needs_permute
        # self.clip = clip_text_module auto-registers the ENTIRE frozen
        # backbone as a submodule, so learner.parameters() would otherwise
        # include all of its weights too -- explicitly freeze them so
        # gradients still flow through to ctx, but the backbone itself is
        # never updated (train_and_eval also restricts the optimizer to
        # [self.ctx] directly, belt and suspenders).
        for p in self.clip.parameters():
            p.requires_grad_(False)
        dtype = clip_text_module.token_embedding.weight.dtype
        ctx_dim = clip_text_module.token_embedding.weight.shape[1]

        prompt_prefix = ' '.join(['X'] * n_ctx)
        prompts = [f'{prompt_prefix} {name}.' for name in class_names]
        tokenized = tokenizer(prompts).to(device)  # (n_cls, context_length)
        with torch.no_grad():
            embedding = clip_text_module.token_embedding(tokenized).type(dtype)

        self.register_buffer('token_prefix', embedding[:, :1, :])       # start token
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])  # classname + eot + pad
        self.register_buffer('tokenized_prompts', tokenized)            # for eot position lookup
        self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device))
        nn.init.normal_(self.ctx, std=0.02)
        self.n_cls = len(class_names)
        self.n_ctx = n_ctx

    def build_prompts(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

    def forward(self):
        x = self.build_prompts() + self.clip.positional_embedding.type(x_dtype(self.clip))
        # Causal masking matters either way -- omitting it was the actual
        # bug: without it, the EOT position attends across all padding
        # positions too (which it never does under causal masking, since
        # padding comes after EOT in sequence order), collapsing to a
        # content-insensitive representation regardless of which class was
        # tokenized.
        if self.needs_permute:
            # Vendored clip.py's Transformer bakes its causal mask in at
            # construction time (inside each ResidualAttentionBlock) and
            # its forward() only takes x -- no attn_mask kwarg to pass.
            x = x.permute(1, 0, 2)
            x = self.clip.transformer(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = self.clip.ln_final(x).type(x_dtype(self.clip))
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)]
        # text_projection is an nn.Parameter (matrix) on the classic clip.py
        # style and ULIP-2/OpenShape's open_clip CLIP, but an nn.Linear on
        # open_clip's newer TextTransformer (Uni3D) -- handle both.
        if isinstance(self.clip.text_projection, nn.Linear):
            x = self.clip.text_projection(x)
        else:
            x = x @ self.clip.text_projection
        x = x / x.norm(dim=-1, keepdim=True)
        return x


def x_dtype(clip_text_module):
    return clip_text_module.token_embedding.weight.dtype


# ---------------------------------------------------------------------------
# Family 3: ImageBind (Point-Bind)
# ---------------------------------------------------------------------------

class ImageBindPromptLearner(nn.Module):
    def __init__(self, model, tokenizer, class_names, n_ctx=N_CTX, device='cuda'):
        super().__init__()
        from imagebind.imagebind_model import ModalityType
        # pointbind_zeroshot_lib.py's POINTBIND wrapper stores the actual
        # ImageBind instance as .bind (self.bind = imagebind_model.imagebind_huge()),
        # not on the wrapper itself -- modality_preprocessors/trunks/heads
        # all live on model.bind, confirmed by how encode_class_text calls
        # model.bind({ModalityType.TEXT: tokens}).
        self.bind = model.bind
        self.ModalityType = ModalityType
        # Same freeze as CLIPStylePromptLearner -- self.bind registers the
        # entire ImageBind backbone as a submodule.
        for p in self.bind.parameters():
            p.requires_grad_(False)
        prep = self.bind.modality_preprocessors[ModalityType.TEXT]
        dtype = prep.token_embedding.weight.dtype
        ctx_dim = prep.token_embedding.weight.shape[1]

        prompt_prefix = ' '.join(['X'] * n_ctx)
        prompts = [f'{prompt_prefix} {name}.' for name in class_names]
        tokenized = torch.cat(
            [tokenizer(p).unsqueeze(0) for p in prompts], dim=0).to(device)  # (n_cls, context_length)
        with torch.no_grad():
            embedding = prep.token_embedding(tokenized).type(dtype)

        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])
        self.register_buffer('tokenized_prompts', tokenized)
        self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device))
        nn.init.normal_(self.ctx, std=0.02)
        self.n_cls = len(class_names)
        self.pos_embed = prep.pos_embed
        # ImageBind's TextPreprocessor is built with causal_masking=True
        # (confirmed in imagebind_model.py) and normally adds its own
        # causal `mask` buffer to trunk_inputs via its own forward() -- since
        # we bypass that forward() to splice in the learned context, we have
        # to replicate this ourselves, or EOT pooling attends across all
        # padding positions and collapses to a class-insensitive output
        # (exactly the bug hit and fixed in CLIPStylePromptLearner).
        self.attn_mask = prep.mask

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        tokens = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        tokens = tokens + self.pos_embed
        trunk_inputs = {'tokens': tokens, 'attn_mask': self.attn_mask}
        text_lengths = self.tokenized_prompts.argmax(dim=-1)
        head_inputs = {'seq_len': text_lengths}
        x = self.bind.modality_trunks[self.ModalityType.TEXT](**trunk_inputs)
        x = self.bind.modality_heads[self.ModalityType.TEXT](x, **head_inputs)
        x = self.bind.modality_postprocessors[self.ModalityType.TEXT](x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

def build_prompt_learner(model_key, lib, class_names, device):
    if model_key in OPEN_CLIP_FAMILY | OPENAI_CLIP_FAMILY:
        built = lib.build_model(device=device)
        if model_key == 'openshape':
            _model, clip_model, tokenizer = built
            text_module, text_tokenizer = clip_model, tokenizer
        elif model_key in {'uni3d'}:
            # Uni3D's open_clip model is a CustomTextCLIP, whose
            # encode_text just delegates to a separate TextTransformer
            # submodule (.text) -- that submodule has the standard
            # token_embedding/positional_embedding/transformer/ln_final/
            # text_projection attributes directly on itself, the CLIP
            # wrapper around it does not.
            text_module, text_tokenizer = built['clip_model'].text, built['tokenizer']
        elif model_key == 'ulip2':
            # ULIP2_WITH_OPENCLIP wraps the real open_clip text tower as
            # .open_clip_model -- it doesn't expose token_embedding itself.
            text_module, text_tokenizer = built.open_clip_model, built.tokenizer
        elif model_key == 'jm3d':
            # ULIP_PointBERT (a different base class, ULIP_WITH_IMAGE) DOES
            # expose token_embedding/transformer/etc directly on itself.
            text_module, text_tokenizer = built, built.tokenizer
        elif model_key == 'pointclipv2':
            clip_model = built['clip_model']
            text_module, text_tokenizer = clip_model, built['tokenizer']
        else:
            raise ValueError(model_key)
        needs_permute = model_key in OPENAI_CLIP_FAMILY
        return lambda cnames: CLIPStylePromptLearner(
            text_module, text_tokenizer, cnames, device=device, needs_permute=needs_permute)
    elif model_key in IMAGEBIND_FAMILY:
        built = lib.build_model(device=device)
        tokenizer = built.tokenizer
        return lambda cnames: ImageBindPromptLearner(built, tokenizer, cnames, device=device)
    raise ValueError(f'Unknown model family for {model_key!r}')


def train_and_eval(prompt_learner_factory, feats, gt, class_names, support_idx, query_idx, device):
    learner = prompt_learner_factory(class_names).to(device)
    optimizer = torch.optim.Adam([learner.ctx], lr=LR)

    support_feats = feats[support_idx]
    support_labels = torch.from_numpy(gt[support_idx]).long().to(device)

    learner.train()
    for _epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        text_features = learner()  # (n_cls, dim), differentiable w.r.t. ctx
        logits = support_feats @ text_features.t()
        loss = F.cross_entropy(logits, support_labels)
        loss.backward()
        optimizer.step()

    learner.eval()
    with torch.no_grad():
        text_features = learner()
        query_feats = feats[query_idx]
        query_logits = query_feats @ text_features.t()
        preds = query_logits.argmax(dim=1).cpu().numpy()
    return balanced_accuracy_score(gt[query_idx], preds), loss.item()


def train_and_get_logits(prompt_learner_factory, feats, gt, class_names, support_idx, query_idx, device):
    """Same training loop as train_and_eval, but returns the raw query
    logits (for combined_fix.py to fuse with another method's logits)
    instead of just the scalar accuracy."""
    learner = prompt_learner_factory(class_names).to(device)
    optimizer = torch.optim.Adam([learner.ctx], lr=LR)

    support_feats = feats[support_idx]
    support_labels = torch.from_numpy(gt[support_idx]).long().to(device)

    learner.train()
    for _epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        text_features = learner()
        logits = support_feats @ text_features.t()
        loss = F.cross_entropy(logits, support_labels)
        loss.backward()
        optimizer.step()

    learner.eval()
    with torch.no_grad():
        text_features = learner()
        query_feats = feats[query_idx]
        query_logits = (query_feats @ text_features.t()).cpu().numpy()
    return query_logits


def run_task(prompt_learner_factory, task_name, task, device='cuda'):
    model_key = prompt_learner_factory.model_key
    cache_path = FEATURE_CACHE_ROOT / f'{task_name}__{model_key}__clean.npz'
    d = np.load(cache_path, allow_pickle=True)
    feats = torch.from_numpy(d['feats']).float().to(device)
    gt = d['gt']
    n_samples = len(gt)
    n_classes = len(task['class_names'])

    rows = []
    for k in K_SHOTS:
        for seed in SEEDS:
            support_idx = stratified_kshot_indices(gt, k, n_classes, seed)
            if support_idx is None:
                continue
            query_mask = np.ones(n_samples, dtype=bool)
            query_mask[support_idx] = False
            query_idx = np.where(query_mask)[0]

            bal_acc, final_loss = train_and_eval(
                prompt_learner_factory, feats, gt, task['class_names'],
                support_idx, query_idx, device)

            rows.append(dict(task=task_name, k=k, seed=seed, n_samples=n_samples,
                              n_classes=n_classes, n_query=len(query_idx),
                              soft_prompt_bal_acc=bal_acc, final_train_loss=final_loss))
    return rows


def run_model(model_key, device='cuda', task_filter=None):
    lib = importlib.import_module(MODEL_LIB[model_key])
    print(f'[{model_key}] preparing prompt-learner factory...')
    all_rows = []
    tasks = ({task_filter: td.ALL_TASKS[task_filter]} if task_filter
             else td.ALL_TASKS)
    for task_name, task in tasks.items():
        factory = build_prompt_learner(model_key, lib, task['class_names'], device)
        factory.model_key = model_key
        rows = run_task(factory, task_name, task, device=device)
        for r in rows:
            r['model'] = model_key
        all_rows.extend(rows)
        # build_prompt_learner rebuilds the full backbone from scratch for
        # every task rather than once per model -- for the largest text
        # towers (ULIP-2/OpenShape's ViT-bigG-14), the previous task's model
        # wasn't being freed before the next one loaded, causing a CUDA OOM
        # on the 2nd task even on an 80GB GPU's smaller a16 sibling. Force
        # release here between tasks.
        del factory
        torch.cuda.empty_cache()
        if rows:
            by_k = {}
            for r in rows:
                by_k.setdefault(r['k'], []).append(r['soft_prompt_bal_acc'])
            summary = ', '.join(f'K={k}:{np.mean(v):.3f}' for k, v in sorted(by_k.items()))
            print(f'[{model_key}/{task_name}] {summary}')
        else:
            print(f'[{model_key}/{task_name}] SKIPPED (not enough samples/class)')
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                     choices=list(MODEL_LIB.keys()) + ['all'])
    ap.add_argument('--task', default=None, choices=list(td.ALL_TASKS.keys()),
                     help='Limit to one task (smoke-testing only).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = list(MODEL_LIB.keys()) if args.model == 'all' else [args.model]

    fieldnames = ['model', 'task', 'k', 'seed', 'n_samples', 'n_classes', 'n_query',
                  'soft_prompt_bal_acc', 'final_train_loss']
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for m in models:
        print('=' * 70)
        print(f'>>> model={m}')
        print('=' * 70)
        rows = run_model(m, device=device, task_filter=args.task)
        out_path = OUT_DIR / f'{m}.csv'
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {len(rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()
