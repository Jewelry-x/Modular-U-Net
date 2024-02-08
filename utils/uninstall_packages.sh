#!/bin/bash
cd ..

# Activate virtual environment
source cath/bin/activate

# Uninstall packages
pip freeze > packages.txt
FOR /F %i IN (packages.txt) DO pip uninstall -y %i

# Deactivate virtual environment
source cath/bin/deactivate.bat

# Delete virtual environment
rm -rf cath