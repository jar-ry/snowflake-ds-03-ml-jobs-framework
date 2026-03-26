from snowflake.ml.jobs import submit_directory


def run(session, conf: dict):
    """Submit an HPO training job to a compute pool via submit_directory."""
    print("=" * 60)
    print("TRAINING PIPELINE (submit_directory)")
    print("=" * 60)

    compute = conf["compute"]

    print(f"Submitting ML Job to pool '{compute['pool_name']}'")
    print("  Entrypoint : src/modelling/train.py")
    print("  Config     : conf/parameters.yml (loaded in container)")
    print(f"  Trials     : {compute['num_trials']}")

    job = submit_directory(
        "./",
        compute["pool_name"],
        entrypoint="src/modelling/train.py",
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
