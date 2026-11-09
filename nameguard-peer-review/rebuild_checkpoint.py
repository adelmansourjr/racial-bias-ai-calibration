"""Recreate the frozen 4-bit derivative in a new directory; never write source files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import bootstrap
from nameguard.mlx_backend import file_digest,parameter_digest,PARAMETER_DIGEST
ROOT=Path(__file__).resolve().parent
SOURCE_REPOSITORY='Qwen/Qwen2.5-1.5B-Instruct'
SOURCE_REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def snapshot(path):
    return {str(p.relative_to(path)):file_digest(p) for p in sorted(path.rglob('*')) if p.is_file()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();source=args.source.resolve();output=args.output.resolve()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    if output.exists() or args.report.exists():raise FileExistsError('Fresh output and report required')
    if source in output.parents or output in source.parents:raise ValueError('Source/output ancestry is forbidden')
    original=snapshot(source)
    config=json.loads((source/'config.json').read_text())
    if config.get('quantization') or config.get('quantization_config'):raise ValueError('Unquantized source checkpoint required')
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.utils import save
    from mlx.utils import tree_flatten
    model,tok=load(str(source),lazy=True)
    for _,value in tree_flatten(model.parameters()):mx.eval(value)
    nn.quantize(model,bits=4,group_size=64);mx.eval(model.parameters())
    config['quantization']={'bits':4,'group_size':64,'mode':'affine'}
    save(str(output),str(source),model,tok,config)
    # mlx_lm.utils.save donates the original arrays. Verify the serialized,
    # reloaded checkpoint instead of inspecting the consumed model instance.
    del model
    model,_=load(str(output),lazy=True)
    digest=parameter_digest(model)
    expected=json.loads((ROOT/'audit/checkpoint-manifest.json').read_text())['files']
    rebuilt=snapshot(output)
    expected_files={key:value['sha256'] for key,value in expected.items()}
    source_unchanged=snapshot(source)==original
    report={'source_repository':SOURCE_REPOSITORY,'source_revision':SOURCE_REVISION,
            'parameter_digest':digest,'matches_pinned_parameters':digest==PARAMETER_DIGEST,
            'matches_every_pinned_checkpoint_file':rebuilt==expected_files,
            'source_files_unchanged':source_unchanged,'reconstructed_files_sha256':rebuilt,
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Same-host reconstruction, not independent external replication. Acquire the declared source revision separately.'}
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not source_unchanged or digest!=PARAMETER_DIGEST:raise RuntimeError('Checkpoint reconstruction failed the parameter/source check')


if __name__=='__main__':main()
