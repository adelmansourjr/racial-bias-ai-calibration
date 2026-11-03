# Exact tested model provenance

The study uses Qwen2.5-1.5B-Instruct from source revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with a separate local MLX affine-four-bit
derivative (group size 64). The measured hidden width is 1,536, with 28 blocks,
12 query heads, two KV heads and tied embeddings. The new hardening study loads
that already-created derivative; it does not convert, adapt or merge weights.

The original local conversion recipe used the independent BF16 copy and the
then-tested MLX runtime:

```python
model, tokenizer = mlx_lm.load(BF16_COPY, lazy=True)
mlx.nn.quantize(model, bits=4, group_size=64)
config["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine"}
mlx_lm.utils.save(NEW_OUTPUT_DIRECTORY, BF16_COPY, model, tokenizer, config)
```

This conversion is not re-executed in the hardening experiment. Model/file
hashes can differ with runtime versions or export metadata; verify the exact
loaded parameter digest, not only a model name. The supported digest is:

```
c6c622310d66489a5a9de5f438c23063bf784b3031f2e8af64bfd60da0394b40
```

`data/tested-checkpoint-manifest.json` contains the actual tested checkpoint
file hashes, with its private local directory omitted. The trusted loader
checks every file and rejects extra/missing files. Its parameter check also
rejects a different quantized tensor checkpoint even if an operator supplies
a new file manifest. This intentional restriction limits cross-model claims.

The original source weights, independent BF16 clone and frozen four-bit
derivative remain local. No weight tensors are bundled or uploaded to GitHub.
Model/tokenizer/dependency licenses remain with their respective owners.
The hardening paper does not claim accuracy parity between four-bit and BF16,
nor between scalarwise and native whole-string input tokenization.
