#!/usr/bin/env bash
python -m venv venv
source venv/bin/activate || source venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "NOTE: torch-geometric may need special install steps depending on your CUDA / torch version."
