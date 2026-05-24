from __future__ import annotations

from sdk.api import ensure_dir, write_json


def main(context, config) -> None:
    ensure_dir(context.output_dir / "predictions")

    # The hosted backend owns hidden test inference. This submission writes the
    # selected config for the backend to apply after the train/val phase.
    write_json(context.output_dir / "selected_config.json", dict(config))
    write_json(
        context.output_dir / "search_results.json",
        {
            "mode": "direct_config",
            "selected": dict(config),
            "trials": [],
        },
    )
    print("[submission] using uploaded config directly; backend will run hidden test inference", flush=True)
