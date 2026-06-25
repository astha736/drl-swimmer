"""Project-level metrics and result persistence helpers."""

from pathlib import Path
from typing import Dict, Optional, Tuple

from matplotlib.backends.backend_pdf import PdfPages

from utils import farms_compat


def get_performance_metrics(
    sim,
    timestep: float,
    n_iterations: int,
    do_plots: bool = False,
    additionalMetrics: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """Collect standard FARMS performance metrics from a completed simulation."""
    sensors = sim.task.data.sensors
    metrics = {}
    plots = {}

    metrics.update(
        farms_compat.link_performance_metrics(
            sensors.links,
            timestep=timestep,
            n_iterations=n_iterations,
        )
    )
    metrics.update(
        farms_compat.joint_performance_metrics(
            sensors.joints,
            timestep=timestep,
            n_iterations=n_iterations,
        )
    )

    if additionalMetrics:
        metrics.update(additionalMetrics)

    # Plot generation was optional in the old helper. Keep the return shape so
    # callers that save plots continue to work, while metrics remain available
    # even when no plot objects are created.
    if not do_plots:
        return metrics, plots
    return metrics, plots


def save_performance_metrics(metrics: Dict, plots: Dict) -> None:
    """Write metrics and optional matplotlib figures to the active result folder."""
    import conf

    result_dir = Path(conf.LOG_DIR_RESULTS)
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = result_dir / "single_test_env_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for key in sorted(metrics):
            metrics_file.write(f"{key}: {metrics[key]}\n")

    if not plots:
        return

    plots_path = result_dir / "single_test_env_plots.pdf"
    with PdfPages(plots_path) as pdf:
        for figure in plots.values():
            pdf.savefig(figure)
