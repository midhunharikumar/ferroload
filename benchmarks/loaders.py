"""Loader adapters — each yields stacked uint8 batches [B,H,W,C].

- hf_arrow   : HF `datasets` (Arrow) + torch DataLoader, PIL decode in workers.
- webdataset : wds.WebLoader over tar shards, PIL decode in workers.
- ferroload  : native make_loader (Rust parallel decode, in-process rayon).

All decode the SAME encoded bytes to the SAME target, so only format+loader differ.
The decoder differs by design: HF/wds use PIL; Ferroload uses its Rust codec — that
IS ferroload's lever, and is called out explicitly in the report.
"""
import io
import functools

import numpy as np
from PIL import Image


# ----- module-level transforms (picklable / fork-safe) -----

def _resize_pil(im, target):
    im = im.convert("RGB")
    if target is not None and im.size != (target, target):
        im = im.resize((target, target), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def hf_collate(samples, target):
    return np.stack([_resize_pil(s["image"], target) for s in samples])


_IMG_KEYS = ("jpg", "jpeg", "png", "webp")


def wds_decode(sample, target):
    for k in _IMG_KEYS:
        if k in sample:
            return _resize_pil(Image.open(io.BytesIO(sample[k])), target)
    raise KeyError(f"no image in {list(sample)}")


# ----- adapters -----

def hf_arrow_loader(arrow_dir, target, batch_size, num_workers):
    import datasets as hfds
    import torch.utils.data as tud
    ds = hfds.load_from_disk(arrow_dir)
    return tud.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        drop_last=True, collate_fn=functools.partial(hf_collate, target=target),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )


def webdataset_loader(shards, target, batch_size, num_workers):
    import webdataset as wds
    ds = (
        wds.WebDataset(shards, shardshuffle=False, empty_check=False)
        .map(functools.partial(wds_decode, target=target))
        .batched(batch_size, collation_fn=np.stack, partial=False)
    )
    return wds.WebLoader(ds, batch_size=None, num_workers=num_workers,
                         prefetch_factor=(4 if num_workers > 0 else None))


def ferroload_native_loader(ds_dir, target, batch_size, prefetch=3):
    import ferroload
    resize = (target, target) if target else None
    dl = ferroload.make_loader(ds_dir, batch_size=batch_size, images=["image"],
                               resize=resize, out="numpy", shuffle=False,
                               prefetch=prefetch)

    def gen():
        for batch in dl:
            yield batch["image"]
    return gen()


class _FerroRawDS:
    """Map-style view returning RAW image bytes from a Ferroload dataset, so a
    torch DataLoader can decode them with PIL in workers — isolates the format
    from the decoder choice."""
    def __init__(self, ds_dir):
        from ferroload._core import Dataset
        self.ds = Dataset.open(ds_dir)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds.read(i, "image")  # raw encoded bytes


def ferro_pil_collate(byte_list, target):
    return np.stack([_resize_pil(Image.open(io.BytesIO(b)), target) for b in byte_list])


def ferro_pil_loader(ds_dir, target, batch_size, num_workers):
    """Ferroload format + PIL decode in worker processes (decoder-isolation run)."""
    import torch.utils.data as tud
    return tud.DataLoader(
        _FerroRawDS(ds_dir), batch_size=batch_size, num_workers=num_workers, shuffle=False,
        drop_last=True, collate_fn=functools.partial(ferro_pil_collate, target=target),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )


def batch_n(b):
    return b.shape[0]
