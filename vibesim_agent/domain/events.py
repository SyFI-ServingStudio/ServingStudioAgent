"""Durable managed-job event kinds shared by history and browser projection."""

MANAGED_JOB_EVENTS = frozenset(
    {
        "simulation.requested",
        "simulation.running",
        "analysis.running",
        "experiment.ready",
        "experiment.failed",
        "experiment.interrupted",
        "job.requested",
        "job.running",
        "job.analysis_running",
        "job.ready",
        "job.failed",
        "job.interrupted",
        "job",
    }
)
