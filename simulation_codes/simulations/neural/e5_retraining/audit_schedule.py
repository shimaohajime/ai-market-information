"""Bound audit concurrency while trajectory continuation is running."""
import json,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import engine as e
from run import CFG,name
from audit_baseline import audit

def ready(seed,x):
    p=e.ROOT/'results'/name(seed,x)/'complete.json'
    return p.exists() and json.loads(p.read_text()).get('slow_updates')==12

if __name__=='__main__':
    while not all(ready(s,0.) for s in CFG['seeds']):time.sleep(5)
    todo=[(s,12,r) for r in range(4) for s in CFG['seeds']]
    with ProcessPoolExecutor(max_workers=3) as pool:
        active=set()
        while todo or active:
            limit=3 if all(ready(s,x) for s in CFG['seeds'] for x in CFG['adoption']) else 1
            while todo and len(active)<limit:active.add(pool.submit(audit,*todo.pop(0)))
            done,active=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for f in done:print('DONE',f.result(),flush=True)
