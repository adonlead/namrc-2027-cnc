"""Download and inspect NASA Milling archive; export verified numeric experimental records.
No fabricated labels or predictive metrics. Fail loudly on unsupported structures.
"""
import hashlib, io, json, os, sys, zipfile, datetime, urllib.request
from pathlib import Path
import numpy as np
from scipy.io import loadmat

URL = 'https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip'
OUT = Path('public/api'); OUT.mkdir(parents=True, exist_ok=True)
MAX_DOWNLOAD = 400 * 1024 * 1024

def normalize(x):
    if isinstance(x, np.ndarray):
        if x.dtype.names:
            return [normalize(v) for v in x.reshape(-1)]
        if x.dtype == object:
            return [normalize(v) for v in x.reshape(-1)]
        return x.tolist()
    if isinstance(x, np.void):
        return {k: normalize(x[k]) for k in x.dtype.names}
    if hasattr(x, '_fieldnames'):
        return {k: normalize(getattr(x,k)) for k in x._fieldnames}
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, bytes): return x.decode('utf-8','replace')
    if isinstance(x, (list, tuple)): return [normalize(y) for y in x]
    if isinstance(x, dict): return {str(k):normalize(v) for k,v in x.items()}
    return x

def scalar(v):
    while isinstance(v,list) and len(v)==1: v=v[0]
    if isinstance(v,(int,float)) and np.isfinite(v): return float(v)
    return None

def summarize_record(record, index):
    # Retain scalar experimental metadata; never confuse sensor time series with labels.
    r={'experiment_index':index}
    for k,v in record.items():
        value=scalar(v)
        if value is not None: r[k]=value
        elif isinstance(v,str): r[k]=v
        elif isinstance(v,list):
            arr=np.asarray(v)
            if arr.size and arr.size < 2000000 and np.issubdtype(arr.dtype,np.number):
                a=arr.astype(float).ravel(); a=a[np.isfinite(a)]
                if len(a):
                    r[k+'_mean']=float(a.mean()); r[k+'_std']=float(a.std()); r[k+'_rms']=float(np.sqrt(np.mean(a*a)))
                    r[k+'_samples']=int(len(a))
    return r

def main():
    print('Downloading NASA Milling archive...')
    req=urllib.request.Request(URL, headers={'User-Agent':'NAMRC-2027-academic-research/0.3'})
    with urllib.request.urlopen(req,timeout=120) as response:
        raw=response.read(MAX_DOWNLOAD+1)
    if len(raw)>MAX_DOWNLOAD: raise RuntimeError('NASA archive exceeds configured size cap')
    sha=hashlib.sha256(raw).hexdigest()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names=archive.namelist()
        mats=[n for n in names if n.lower().endswith('.mat') and not n.startswith('__MACOSX/')]
        if not mats: raise RuntimeError('No MATLAB .mat files found; inspect NASA archive format')
        records=[]; mat_inventory=[]; diagnostics=[]
        for name in mats:
            if archive.getinfo(name).file_size>500*1024*1024:
                diagnostics.append(name+': uncompressed size too large');continue
            try:
                obj=loadmat(io.BytesIO(archive.read(name)),simplify_cells=True)
                keys=[k for k in obj if not k.startswith('__')]
                mat_inventory.append({'file':name,'variables':keys})
                for key in keys:
                    value=normalize(obj[key])
                    candidates=value if isinstance(value,list) else [value]
                    for candidate in candidates:
                        if isinstance(candidate,dict):
                            rec=summarize_record(candidate,len(records)+1)
                            rec['source_file']=name;rec['source_variable']=key
                            if len(rec)>3: records.append(rec)
            except NotImplementedError:
                diagnostics.append(name+': MATLAB v7.3/HDF5 unsupported by scipy; requires h5py adapter')
            except Exception as exc:
                diagnostics.append(name+': '+str(exc)[:300])
    if not records: raise RuntimeError('No structured experiment records parsed. Review archive layout and parser. '+str(diagnostics))
    # Preserve only parsed measurements, without invented predictions.
    generated=datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest={'dataset':'NASA Milling Wear','source_url':URL,'sha256':sha,'retrieved_utc':generated,
              'archive_bytes':len(raw),'archive_files':names,'mat_inventory':mat_inventory,
              'records':len(records),'warnings':diagnostics,
              'research_notice':'Historical experiments; feature summaries are descriptive, not validated predictions.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'experiments.json').write_text(json.dumps({'metadata':{'sha256':sha,'generated_utc':generated},'records':records},allow_nan=False),encoding='utf-8')
    print('Wrote',len(records),'experimental records; SHA256',sha)

if __name__=='__main__': main()
