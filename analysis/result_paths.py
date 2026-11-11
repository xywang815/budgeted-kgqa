"""Resolve public workflow directories and legacy local analysis directories."""
from pathlib import Path

ALIASES = {'答案来源分析': 'provenance', '机制核验': 'mechanism',
           '独立确认分析': 'confirmation_analysis'}

def result_file(root, group, filename):
    root = Path(root)
    candidates = [root / ALIASES.get(group, group) / filename, root / group / filename]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('Expected analysis output: ' + ' or '.join(map(str, candidates)))
