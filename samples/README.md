# Samples

End-to-end example training code built on the `ferroload` Python package
(`pip install ferroload`). Unlike `benchmarks/` (throughput measurements) and
`python/` (dev/test scripts), these are complete, runnable training programs.

| Sample | What it shows |
|---|---|
| [`image_text_determinism/`](image_text_determinism/) | Mini-CLIP contrastive training on Flickr30k (image + caption pairs) with **bitwise-identical loss curves** across independent runs — the loader's deterministic sampling, end to end. Runs on [Modal](https://modal.com). |
