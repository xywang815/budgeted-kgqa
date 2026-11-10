#!/usr/bin/env python3
"""Create an empty execution workspace from code; never copy experiment data."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'work/reproduction')
    args = ap.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise SystemExit(f'Refusing to overwrite an existing workspace: {dest}')
    dest.mkdir(parents=True)
    for name in ('src', 'supplement', 'scripts'):
        for source in (ROOT / name).rglob('*'):
            if source.is_file() and source.suffix in ('.py', '.sh', '.md') and not set(source.relative_to(ROOT / name).parts) & {'inputs','results','models','outputs','__pycache__'}:
                target = dest / name / source.relative_to(ROOT / name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    # The original shared-pool runner resolves E5 and its protocol here.
    shutil.copytree(dest / 'supplement', dest / 'revision_20260909')
    for source in (ROOT / 'analysis/original').iterdir():
        if source.suffix in ('.py', '.sh'):
            shutil.copy2(source, dest / source.name)
    for name in ('results', 'derived', 'manuscript/figures', 'logs', 'data',
                 'supplement/results', 'audit', 'models', 'revision_20260909/models'):
        (dest / name).mkdir(parents=True, exist_ok=True)
    print(f'Created code-only workspace: {dest}')

if __name__ == '__main__':
    main()
