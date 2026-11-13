from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent;out=root.parent/'output/s4_slow_fast_retraining_v1';out.mkdir(parents=True,exist_ok=True)
complete={str(p.parent.relative_to(root)) for p in (root/'results').glob('*/round*/complete.json')}
(root/'PACKAGING_STATUS.json').write_text(json.dumps(dict(completed_slow_states=len(complete),stages=sorted(complete)),indent=2))
files=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
 r=p.relative_to(root)
 if r.parts[0] in ['logs','data'] or p.name.startswith('report_review-') or p.name=='MANIFEST_SHA256.json':continue
 if r.parts[0]=='results' and len(r.parts)>3 and str(Path(*r.parts[:3])) not in complete:continue
 files.append(p)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2));files.append(root/'MANIFEST_SHA256.json')
archive=out/'AI_Trading_Slow_Fast_Retraining_v1_20261028.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,str(Path(root.name)/p.relative_to(root)))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,h in manifest.items():assert hashlib.sha256(z.read(root.name+'/'+name)).hexdigest()==h
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,files=len(files),completed_slow_states=len(complete)),indent=2))
