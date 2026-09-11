"""Register exact Analyzer evidence discovered during a managed turn."""

from collections.abc import Callable

from ..domain.evidence import (
    build_aggregate_citation_dictionary,
    build_kernel_measurement_citation_dictionary,
    build_kernel_profile_citation_dictionary,
    build_prediction_citation_dictionary,
    build_run_citation_dictionary,
    latest_citation_dictionary,
    merge_citation_dictionaries,
)
from ..storage.jobs import Jobs
from .capabilities import Capability
from .turn import TurnService


class CitationConflict(ValueError):
    pass


class CitationService:
    def __init__(self, turns: TurnService, jobs: Callable[[str], Jobs]):
        self.turns = turns
        self.jobs = jobs

    def register(
        self,
        capability: Capability,
        *,
        resource_kind: str = "aggregate",
        experiment_id: str | None = None,
        prediction_id: str | None = None,
        run_id: str | None = None,
        profile_id: str | None = None,
        measurement_id: str | None = None,
        resource_path: str | None = None,
        analysis: dict | None = None,
    ) -> dict:
        store = self.turns.storage(capability.workspace_id)
        turn = store.turns.get(capability.conversation_id, capability.turn_id)
        if turn is None:
            raise KeyError("managed turn not found")
        if turn["status"] != "running":
            raise CitationConflict("managed turn has already ended")
        if resource_kind == "run":
            if run_id is None or resource_path is None:
                raise ValueError(
                    "run citation registration requires runId and resourcePath"
                )
            dictionary = build_run_citation_dictionary(
                workspace_id=capability.workspace_id,
                run_id=run_id,
                resource_path=resource_path,
            )
            resource_id, identity_key = run_id, "runId"
        elif resource_kind == "prediction":
            if prediction_id is None or resource_path is None:
                raise ValueError(
                    "prediction citation registration requires predictionId and resourcePath"
                )
            dictionary = build_prediction_citation_dictionary(
                workspace_id=capability.workspace_id,
                prediction_id=prediction_id,
                resource_path=resource_path,
            )
            resource_id, identity_key = prediction_id, "predictionId"
        elif resource_kind == "kernel_profile":
            if profile_id is None or resource_path is None or analysis is None:
                raise ValueError(
                    "kernel profile citation registration requires profileId, resourcePath, and analysis"
                )
            dictionary = build_kernel_profile_citation_dictionary(
                workspace_id=capability.workspace_id,
                profile_id=profile_id,
                resource_path=resource_path,
                analysis=analysis,
            )
            resource_id, identity_key = profile_id, "profileId"
        elif resource_kind == "kernel_measurement":
            if measurement_id is None or resource_path is None or analysis is None:
                raise ValueError(
                    "kernel measurement citation registration requires measurementId, resourcePath, and analysis"
                )
            dictionary = build_kernel_measurement_citation_dictionary(
                workspace_id=capability.workspace_id,
                measurement_id=measurement_id,
                resource_path=resource_path,
                analysis=analysis,
            )
            resource_id, identity_key = measurement_id, "measurementId"
        elif resource_kind == "aggregate":
            if experiment_id is None or analysis is None:
                raise ValueError(
                    "aggregate citation registration requires experimentId and analysis"
                )
            experiment = self.jobs(capability.workspace_id).get_experiment(
                experiment_id
            )
            if experiment is not None and experiment["status"] != "ready":
                raise CitationConflict("Analyzer experiment is not ready")
            dictionary = build_aggregate_citation_dictionary(
                analysis,
                workspace_id=capability.workspace_id,
                experiment_id=experiment_id,
            )
            resource_id, identity_key = experiment_id, "experimentId"
        else:
            raise ValueError("unsupported citation resource kind")
        registered_entries = [
            entry.model_dump(by_alias=True) for entry in dictionary.entries
        ]
        events = store.turns.events(capability.conversation_id, capability.turn_id)
        dictionary = merge_citation_dictionaries(
            latest_citation_dictionary(events), dictionary
        )
        snapshot = dictionary.model_dump(by_alias=True)
        payload = {
            "resourceKind": resource_kind,
            "resourceId": resource_id,
            identity_key: resource_id,
            "dictionary": snapshot,
        }
        try:
            store.turns.append_event(capability.turn_id, "citation.dictionary", payload)
        except ValueError as error:
            raise CitationConflict(str(error)) from error
        return {**snapshot, "registeredEntries": registered_entries}
