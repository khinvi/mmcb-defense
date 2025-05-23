import sys
import os
import pytest
import subprocess
sys.path.insert(0, os.path.abspath('src'))
from attacks.generator import FileBasedAttackGenerator

def test_attack_generators():
    gen = FileBasedAttackGenerator(output_dir='data/generated/test_attacks')
    types = ['json', 'csv', 'yaml', 'xml', 'python', 'javascript']
    for t in types:
        # Benign
        path = gen.generate_attack(t, variant='benign', filename=f'benign_{t}_test')
        assert os.path.exists(path), f"Benign {t} file not created!"
        assert os.path.getsize(path) > 0, f"Benign {t} file is empty!"
        # Injection (where supported)
        if t in ['json', 'csv', 'yaml', 'xml']:
            path = gen.generate_attack(t, variant='injection', malicious_instruction='TEST', filename=f'inject_{t}_test')
            assert os.path.exists(path), f"Injected {t} file not created!"
            assert os.path.getsize(path) > 0, f"Injected {t} file is empty!"
        elif t in ['python', 'javascript']:
            path = gen.generate_attack(t, variant='comment_injection', malicious_instruction='TEST', filename=f'inject_{t}_test')
            assert os.path.exists(path), f"Injected {t} file not created!"
            assert os.path.getsize(path) > 0, f"Injected {t} file is empty!"
    # Cleanup
    import shutil
    shutil.rmtree('data/generated/test_attacks', ignore_errors=True)

def test_generate_mmcb_examples_script_runs():
    result = subprocess.run([sys.executable, 'src/attacks/generate_mmcb_examples.py'], capture_output=True)
    assert result.returncode == 0, f"generate_mmcb_examples.py failed: {result.stderr.decode()}"
