import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
tabm_model.py -- re-exports TabM from 04_tabm_model.py
This alias file allows other scripts to use:
    from tabm_model import TabM
without dealing with the numbered filename.
"""
# We cannot directly import from a file starting with a digit,
# so we use importlib.
import importlib.util, os, sys

_spec = importlib.util.spec_from_file_location(
    "tabm_model_impl",
    os.path.join(os.path.dirname(__file__), "04_tabm_model.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export
TabM            = _mod.TabM
BatchEnsembleLinear = _mod.BatchEnsembleLinear
TabMBlock       = _mod.TabMBlock
PLEEncoding     = _mod.PLEEncoding
PLEEmbedder     = _mod.PLEEmbedder


