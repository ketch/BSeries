#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import doctest
import nodepy
import unittest
import os
import subprocess
import tempfile
import sys
import nbformat

if sys.version_info >= (3,0):
    kernel = 'python3'
else:
    kernel = 'python2'

def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=120",
                "--ExecutePreprocessor.kernel_name="+kernel,
                "--output", fout.name, path]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.reads(fout.read().decode('utf-8'), nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors

def run_tests():
    for filename in os.listdir('./examples'):
        if (filename.split('.')[-1] == 'ipynb'):
            print('running notebook: '+ filename)
            _, errors = _notebook_run('./examples/'+filename)
            if errors != []:
                raise(Exception)

    for module_name in ['BSeries',
                        'trees']:
        module = nodepy.__getattribute__(module_name)
        doctest.testmod(module)

    unittest.main(module='BSeries.unit_tests',exit=False)

if __name__ == '__main__':
    run_tests()
