#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Updating apt and installing python3-tk..."
    sudo apt update
    sudo apt install -y python3-tk
fi

echo "Installing Python requirements..."
pip install -r requirements.txt