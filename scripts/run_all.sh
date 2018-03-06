#!/bin/bash
set -x
python scripts/subu.py --model linear --level1 --level2 --level3
python scripts/subu.py --model subu --level1 --level2 --level3
python scripts/subu.py --model nn-2 --level1 --level2 --level3
python scripts/subu.py --model nn-4 --level1 --level2 --level3
python scripts/generate_subu.py --model linear --level1 --level2 --level3
python scripts/generate_subu.py --model subu --level1 --level2 --level3
python scripts/generate_subu.py --model nn-2 --level1 --level2 --level3
python scripts/generate_subu.py --model nn-4 --level1 --level2 --level3
