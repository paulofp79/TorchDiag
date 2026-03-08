import json
import time

from torchdiag import TorchDiag


def test_minimal_run_json(tmp_path):
    diag = TorchDiag(job_name="test_run", sample_interval_ms=50, snapshot_interval_steps=1)
    diag.start()
    for step in range(3):
        with diag.step_context(step=step, batch_size=4):
            time.sleep(0.01)
    diag.stop()

    output = tmp_path / "run.json"
    diag.write_json(str(output))

    data = json.loads(output.read_text())
    assert data["metadata"]["job_name"] == "test_run"
    assert len(data["steps"]) == 3
    assert "samples" in data
    assert "findings" in data
