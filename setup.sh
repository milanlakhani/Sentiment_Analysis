#!/bin/bash

# Create a virtual environment
python3 -m venv senmil

# Activate the virtual environment
source senmil/bin/activate

# Install the required packages
pip install -r requirements.txt
