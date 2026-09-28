"""Download the official OULAD archive from its published Figshare record."""
import json,urllib.request,hashlib,zipfile
from pathlib import Path
out=Path('data/raw/oulad');out.mkdir(parents=True,exist_ok=True)
url='https://api.figshare.com/v2/articles/5081998'
meta=json.load(urllib.request.urlopen(url,timeout=60))
(out/'source_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
for f in meta['files']:
 target=out/f['name']
 if not target.exists(): urllib.request.urlretrieve(f['download_url'],target)
 if f.get('computed_md5') and hashlib.md5(target.read_bytes()).hexdigest()!=f['computed_md5']:raise ValueError('Download checksum mismatch')
 if zipfile.is_zipfile(target):
  with zipfile.ZipFile(target) as archive:
   for name in archive.namelist():
    resolved=(out/name).resolve()
    if not resolved.is_relative_to(out.resolve()):raise ValueError('Unsafe archive path')
   archive.extractall(out)
 print(target,flush=True)
