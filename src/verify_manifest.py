from pathlib import Path
import csv, hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'MANIFEST_SHA256.csv'
fail=[]
with manifest.open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        rel=row['path']; p=ROOT/rel
        if not p.is_file():
            fail.append((rel,'MISSING')); continue
        h=hashlib.sha256(p.read_bytes()).hexdigest(); size=p.stat().st_size
        if h != row['sha256'] or size != int(row['bytes']):
            fail.append((rel,'HASH/SIZE MISMATCH'))
if fail:
    for rel,msg in fail: print(f'{rel}: {msg}')
    print(f'MANIFEST VERIFICATION: FAIL ({len(fail)} file(s))')
    sys.exit(1)
print('MANIFEST VERIFICATION: PASS')
