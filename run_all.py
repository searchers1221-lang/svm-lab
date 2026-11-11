"""Запуск всех уровней подряд. Запускать из корня репозитория: python run_all.py"""
import subprocess, sys
for f in ["src/level_a_moons.py", "src/level_b_churn.py", "src/level_c_scaling.py"]:
    print(f"$ python {f}", flush=True)
    subprocess.run([sys.executable, f], check=True)
