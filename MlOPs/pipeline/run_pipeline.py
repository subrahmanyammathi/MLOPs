import subprocess
import sys
from pathlib import Path

pipeline_dir = Path(__file__).resolve().parent
project_dir = pipeline_dir.parent

preprocess_script = project_dir / "preprocess.py"
steps = []

if preprocess_script.is_file():
    steps.append(preprocess_script)
else:
    print("preprocess.py not found; using the existing processed dataset.\n")

steps.extend([
    pipeline_dir / "train.py",
    pipeline_dir / "predict.py",
])

print("Starting ML Pipeline...\n")

for script in steps:
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    print(f"Running {script.name}...")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=project_dir,
        check=True,
    )

print("\nPipeline executed successfully.")