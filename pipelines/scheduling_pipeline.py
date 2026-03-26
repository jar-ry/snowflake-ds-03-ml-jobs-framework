from src.ml_engineering.scheduling import create_inference_procedure, create_inference_task


def run(session, conf: dict):
    """Create a stored procedure and a suspended Snowflake Task for scheduled batch inference."""
    print("=" * 60)
    print("SCHEDULING PIPELINE")
    print("=" * 60)

    print("\n[1/2] Creating batch inference stored procedure...")
    procedure_fqn = create_inference_procedure(session, conf)

    print("[2/2] Creating scheduled task (suspended)...")
    task_name = create_inference_task(session, conf, procedure_fqn)

    print("\nScheduling pipeline complete.")
    print(f"Task '{task_name}' is SUSPENDED by default.")
    print(f"To activate: ALTER TASK {task_name} RESUME;")
    return task_name
