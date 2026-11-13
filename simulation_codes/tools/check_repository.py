#!/usr/bin/env python3
"""Dependency-free structural and accidental-secret preflight for public-repo preparation."""
from pathlib import Path
import re,sys
root=Path(__file__).resolve().parents[1]
required=['README.md','CITATION.cff','paper/main.tex','paper/sections/S7.tex','empirical/README.md','docs/SOURCE_PROVENANCE.md','tools/import_codex_empirical.py']
missing=[x for x in required if not (root/x).exists()]
if missing:
    print('MISSING',missing);sys.exit(2)
prohibited_ext={'.pt','.pth','.ckpt','.npz','.npy','.parquet','.feather','.sqlite','.pkl','.pickle','.ttf','.otf'}
leak_patterns={
    'private-key':re.compile(r'-----BEGIN (?:RSA|OPENSSH|EC|PRIVATE) PRIVATE KEY-----'),
    'api-key':re.compile(r'(?<![a-zA-Z0-9])sk-[a-zA-Z0-9]{20,}'),
    'homepath':re.compile(r'/(?:Users|home)/[A-Za-z0-9._-]+/'),
    'windows-userpath':re.compile(r'C:\\\\Users\\\\[^\\\\]+\\\\',re.I),
    'one-drive':re.compile(r'OneDrive-(?:The)?Pennsylvania',re.I),
}
issues=[];n=0
for f in root.rglob('*'):
    if not f.is_file():continue
    n+=1
    rel=f.relative_to(root)
    if f.suffix.lower() in prohibited_ext:issues.append((str(rel),'excluded binary extension'))
    if any(x in rel.parts for x in ('raw','private','__pycache__')):issues.append((str(rel),'private/raw/cache directory'))
    if f.suffix.lower() in {'.py','.txt','.md','.tex','.json','.yml','.yaml','.toml','.cff','.cfg','.ini','.sh'} or f.name=='.gitignore':
        data=f.read_text(encoding='utf-8',errors='replace')
        for tag,p in leak_patterns.items():
            if p.search(data):issues.append((str(rel),tag))
print(f'Checked {n} files; required files present; {len(issues)} possible sensitive items')
for x in issues[:50]:print('WARN',*x)
sys.exit(1 if issues else 0)
