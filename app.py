"""CardioTwin X Root Launcher."""
import os
import sys

# Ensure cardiotwin is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and execute app from cardiotwin
import cardiotwin.app
