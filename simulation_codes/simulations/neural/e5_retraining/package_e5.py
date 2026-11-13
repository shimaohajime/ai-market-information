from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent
out=root.parent/'output'/'e5_20261029';out.mkdir(parents=True,exist_ok=True)
skip={'__pycache__','data','cache','.pytest_cache','figures'}
obsolete={'make_report.py','package_results.py','PACKAGING_STATUS.json','MANIFEST_SHA256.json','interrupted_run.log'}
files=[p for p in root.rglob('*') if p.is_file() and not any(x in skip for x in p.relative_to(root).parts) and p.name not in obsolete and p.suffix!='.pyc']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
files.append(root/'MANIFEST_SHA256.json')
archive=out/'AI_Trading_E5_Continuation_20261029.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):z.write(p,'e5_continuation/'+str(p.relative_to(root)))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for rel,sha in manifest.items():assert hashlib.sha256(z.read('e5_continuation/'+rel)).hexdigest()==sha
print(archive,archive.stat().st_size,len(files))
