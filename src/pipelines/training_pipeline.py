import os
import shutil
import tempfile

from snowflake.ml.jobs import submit_directory


def _build_payload(project_root: str) -> str:
    """Assemble a clean payload: src/ contents + conf/ from the project root."""
    tmp_dir = tempfile.mkdtemp(prefix="ml_job_payload_")
    payload = os.path.join(tmp_dir, "payload")

    src_dir = os.path.join(project_root, "src")
    shutil.copytree(
        src_dir,
        payload,
        ignore=shutil.ignore_patterns("__pycache__", "pipelines"),
    )

    conf_src = os.path.join(project_root, "conf")
    conf_dst = os.path.join(payload, "conf")
    shutil.copytree(conf_src, conf_dst)

    return payload


def run(session, conf: dict):
    """Submit an HPO training job to a compute pool via submit_directory."""
    print("=" * 60)
    print("TRAINING PIPELINE (submit_directory)")
    print("=" * 60)

    compute = conf["compute"]

    print(f"Submitting ML Job to pool '{compute['pool_name']}'")
    print("  Entrypoint : modelling/train.py")
    print("  Config     : conf/parameters.yml (copied into payload)")
    print(f"  Trials     : {compute['num_trials']}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    payload_dir = _build_payload(project_root)
    try:
        job = submit_directory(
            payload_dir,
            compute["pool_name"],
            entrypoint="modelling/train.py",
            stage_name=compute["stage_name"],
            session=session,
            target_instances=compute.get("target_instances", 1),
        )

        print(f"Job submitted: {job.id}")
        print("Waiting for job completion...")
        job.wait()
        status = job.status
        print(f"Job status: {status}")
        if status != "DONE":
            logs = job.get_logs()
            print(f"\n--- JOB LOGS ---\n{logs}\n--- END LOGS ---")
            raise RuntimeError(f"Training job failed with status: {status}")
        return job
    finally:
        shutil.rmtree(payload_dir, ignore_errors=True)
