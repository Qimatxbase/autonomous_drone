"""Refresh result.ipynb using this repository's local Python kernel."""
import os
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(ROOT/'.venv').resolve():
    raise SystemExit('Run ./run_result.command from this repository.')
os.environ['JUPYTER_PATH']=str(ROOT/'.venv/share/jupyter')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'result/.matplotlib'))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'.venv/.ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'.venv/.jupyter_runtime'))
import nbformat
from nbclient import NotebookClient
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--notebook',default='result.ipynb')
args=parser.parse_args()
path=(ROOT/args.notebook).resolve()
if not path.is_relative_to(ROOT):
    raise ValueError('Notebook must be inside this repository')
notebook=nbformat.read(path,as_version=4)
NotebookClient(notebook,timeout=180,kernel_name='quangn-drone',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(notebook)
temporary=path.with_suffix('.ipynb.tmp');nbformat.write(notebook,temporary);temporary.replace(path)
print('Notebook refreshed:',path)
print('Interpreter:',sys.executable)
