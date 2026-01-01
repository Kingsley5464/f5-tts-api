#!/usr/bin/env bash
set -e

echo "Removing preinstalled torch packages"
pip uninstall -y torch torchaudio torchvision || true

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing requirements"
pip install --no-cache-dir -r requirements.txt
