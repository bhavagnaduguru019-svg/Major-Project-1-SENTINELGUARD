import os
import sys
import subprocess

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING SENTINELGUARD STREAMLIT DASHBOARD")
    print("Dashboard will open in your default browser at http://localhost:8501")
    print("=" * 60)
    app_path = os.path.join(BASE_DIR, "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
