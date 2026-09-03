#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "=== 1. Upgrading Pip and Installing Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== 2. Collecting Static Files with WhiteNoise ==="
python manage.py collectstatic --no-input

echo "=== 3. Running Database Migrations ==="
python manage.py migrate

echo "=== BUILD FINISHED SUCCESSFULLY ==="
