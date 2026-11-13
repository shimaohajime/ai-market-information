from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent;out=root.parent/'output'/'e9e10_20261029';out.mkdir(parents=True,exist_ok=True)
skip={'data','cache','__pycache__','.pytest_cache'}
files=[p for p in root.rglob('*') if p.is_file() and not any(x in skip for x in p.relative_to(root).parts) and p.suffix!='.pyc' and p.name!='MANIFEST_SHA256.json']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(root/'MANIFEST_SHA256.json')
p=out/'AI_Trading_E9_E10_Dynamics_and_Extensions_20261029.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(files):z.write(f,'e9_e10_dynamics/'+str(f.relative_to(root)))
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 for f,sha in manifest.items():assert hashlib.sha256(z.read('e9_e10_dynamics/'+f)).hexdigest()==sha
print(p,p.stat().st_size,len(files))
