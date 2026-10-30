"""Refresh a PPO evaluation notebook using an installed Jupyter kernel."""
import os
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'.cache/ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'.cache/jupyter'))
import nbformat
from nbclient import NotebookClient
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--notebook',default='notebooks/ppo_curriculum_evaluation.ipynb')
parser.add_argument('--kernel',default='python3',help='Installed kernel with the project dependencies')
args=parser.parse_args()
path=(ROOT/args.notebook).resolve()
if not path.is_relative_to(ROOT):
    raise ValueError('Notebook must be inside this repository')
notebook=nbformat.read(path,as_version=4)
NotebookClient(notebook,timeout=180,kernel_name=args.kernel,resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(notebook)
temporary=path.with_suffix('.ipynb.tmp');nbformat.write(notebook,temporary);temporary.replace(path)
print('Notebook refreshed:',path)
print('Interpreter:',sys.executable)
