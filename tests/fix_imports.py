#!/usr/bin/env python3
"""Fix imports in test files."""

import os
import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace import patterns
    patterns = [
        (r'import riemannopt\n\nfrom tests\.conftest import', 'from conftest import riemannopt,'),
        (r'import riemannopt\n\n', 'from conftest import riemannopt\n\n'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed: {filepath}")

# Files to fix
files = [
    'unit/manifolds/test_stiefel.py',
    'unit/manifolds/test_grassmann.py', 
    'unit/optimizers/test_sgd.py',
    'unit/optimizers/test_adam.py',
    'unit/optimizers/test_lbfgs.py',
    'benchmarks/test_performance.py',
    'numerical/test_stability.py',
]

for f in files:
    if os.path.exists(f):
        fix_file(f)