"""Bounded standalone build-context whitelist; no network or credential reads."""
import argparse,hashlib,json,shutil,stat
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--stage',type=Path);args=ap.parse_args()
    source=args.source.resolve();manifest=json.loads((source/'publication-source-manifest.json').read_text(encoding='utf-8'))
    approved=manifest['source_files'];actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() and '.git' not in p.relative_to(source).parts}
    metadata={'README.md','publication-source-manifest.json','verify_public_source.py','.github/workflows/publish.yml'}
    assert actual==set(approved)|metadata,'unexpected public repository files'
    for relative,digest in approved.items():
        path=source/relative;mode=path.lstat().st_mode
        assert stat.S_ISREG(mode) and not path.is_symlink() and path.resolve().is_relative_to(source)
        assert path.stat().st_size<=65536 and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    if args.stage:
        stage=args.stage.resolve();assert not stage.exists(),'build context must be fresh';stage.mkdir(parents=True)
        for relative in approved:
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,target)
        assert {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}==set(approved)
    print(json.dumps({'approved_build_files':len(approved),'private_data_or_extra_files_in_context':False,'source_hashes_verified':True,'context_staged':args.stage is not None}))

if __name__=='__main__':main()
