import json, subprocess, sys

def test_pipeline_runs():
    p=subprocess.run([sys.executable,"-m","ws_lite.cli","reproduce"],capture_output=True,text=True,check=True)
    data=json.loads(p.stdout.strip())
    assert 0.0<=data["ws_classifier_acc"]<=1.0
    assert 0.0<=data["small_gold_classifier_acc"]<=1.0

def test_ablation_runs():
    p=subprocess.run([sys.executable,"-m","ws_lite.cli","ablation"],capture_output=True,text=True,check=True)
    data=json.loads(p.stdout.strip())
    assert "leave_one_out_drop" in data
