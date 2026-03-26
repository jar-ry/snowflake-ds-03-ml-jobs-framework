from snowflake.ml.registry import Registry


def get_best_model_version(
    mr: Registry,
    model_name: str,
    metric: str = "mean_absolute_percentage_error",
    mode: str = "min",
):
    """Find the model version with the best value for the given metric.

    Returns:
        Tuple of (best ModelVersion, best score). Falls back to the latest
        version if no versions have the requested metric.
    """
    model = mr.get_model(model_name)
    versions = model.versions()
    if not versions:
        return None, None

    best_version = None
    best_score = float("inf") if mode == "min" else float("-inf")
    compare = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    for v in versions:
        all_metrics = v.show_metrics()
        if not all_metrics or metric not in all_metrics:
            continue
        score = all_metrics[metric]
        if compare(score, best_score):
            best_score = score
            best_version = v

    if best_version is None:
        print(f"No versions have metric '{metric}'. Falling back to latest version.")
        best_version = versions[-1]
        best_score = None

    return best_version, best_score


def promote_model(session, mr: Registry, model_name: str, version_name: str):
    """Promote a model version by tagging it as live and setting it as the default."""
    db, schema = mr.location.split(".")
    model = mr.get_model(model_name)

    # 1. Create tag if not exists
    tag_fqn = f"{db}.{schema}.live_model_version"
    session.sql(f"CREATE OR REPLACE TAG {tag_fqn};").collect()
    print(f"Tag '{tag_fqn}' created")

    # 2. Set tag on model object with best version name
    model.set_tag(tag_fqn, version_name)
    print(f"Tag '{tag_fqn}' set to '{version_name}' on {model_name}")

    # 3. Set default version
    model.default = version_name
    print(f"Default version set to '{version_name}' for {model_name}")

    return model.version(version_name)
