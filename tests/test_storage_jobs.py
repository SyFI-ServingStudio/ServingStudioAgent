import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import Role
from vibesim_agent.storage.conversations import Conversations
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.jobs import Jobs
from vibesim_agent.storage.turns import Turns


class JobStorageTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.database = Database.create(root / "state.sqlite")
        self.jobs = Jobs(self.database)
        self.turns = Turns(self.database)
        conversations = Conversations(self.database)
        runtime = RoleRuntime("test", "scope", "model", "high", "default")
        for cid in ("c", "other"):
            conversations.create(
                cid, title=cid, runtimes={role: runtime for role in Role}
            )
            self.turns.start(cid, cid + "-turn", "question")

    def simulation(self, **overrides):
        values = {
            "conversation_id": "c",
            "turn_id": "c-turn",
            "role": "assistant",
            "job_id": "j",
            "experiment_id": "e",
            "experiment_path": "run",
            "event": {"kind": "simulation.requested"},
        }
        return self.jobs.create_simulation(**(values | overrides))

    def artifact(self, **overrides):
        values = {
            "conversation_id": "c",
            "turn_id": "c-turn",
            "role": "assistant",
            "job_id": "a",
            "resource_id": "r",
            "job_kind": "timing_predict",
            "artifact_path": "artifact",
            "analyzer_resource_id": "ar",
            "event": {"kind": "job.requested"},
        }
        return self.jobs.create_artifact(**(values | overrides))

    def update(self, job_id="j", **overrides):
        values = {
            "conversation_id": "c",
            "turn_id": "c-turn",
            "status": "completed",
            "simulation": True,
            "event": lambda job: {"kind": "simulation.completed", **job},
        }
        return self.jobs.update(job_id, **(values | overrides))

    def snapshot(self):
        with self.database.connect() as connection:
            return {
                table: [
                    tuple(row) for row in connection.execute(f"SELECT * FROM {table}")
                ]
                for table in (
                    "execution_jobs",
                    "experiments",
                    "conversation_experiments",
                    "turn_events",
                )
            }

    def reject_events(self):
        with self.database.connect(write=True) as connection:
            connection.execute(
                "CREATE TRIGGER reject_event BEFORE INSERT ON turn_events "
                "BEGIN SELECT RAISE(ABORT, 'event rejected'); END"
            )

    def test_simulation_relationship_and_status_event_share_ordered_turn(self):
        self.turns.append_event("c-turn", "role_ready", {"role": "assistant"})
        job = self.simulation()
        self.assertEqual(
            job,
            {
                "job_id": "j",
                "experiment_id": "e",
                "status": "requested",
                "experiment_path": "run",
                "job_kind": "simulation",
            },
        )
        result = self.update()
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("id", result)
        experiment = self.jobs.get_experiment("e")
        self.assertEqual(experiment, self.jobs.experiment_by_path("run"))
        self.assertEqual(
            (experiment["status"], experiment["job_id"]), ("completed", "j")
        )
        link = self.jobs.list_experiments("c")[0]
        self.assertEqual((link["turn_id"], link["relation"]), ("c-turn", "produced"))
        events = self.turns.events("c", "c-turn")
        self.assertEqual([e["sequence"] for e in events], [0, 1, 2])
        self.assertEqual(
            events[-1]["payload"], {"kind": "simulation.completed", **result}
        )

    def test_same_path_keeps_identity_origin_time_and_refreshes_produced_turn(self):
        with patch("vibesim_agent.storage.jobs.time.time", return_value=10):
            self.simulation()
        with self.database.connect(write=True) as connection:
            connection.execute("UPDATE experiments SET origin_kind = 'imported'")
        self.turns.start("c", "next", "again")
        with patch("vibesim_agent.storage.jobs.time.time", return_value=20):
            self.simulation(job_id="j2", turn_id="next")
        experiment = self.jobs.get_experiment("e")
        self.assertEqual(
            (
                experiment["origin_kind"],
                experiment["created_at"],
                experiment["updated_at"],
                experiment["job_id"],
            ),
            ("imported", 10, 20, "j2"),
        )
        self.assertEqual(self.jobs.list_experiments("c")[0]["turn_id"], "next")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.simulation(job_id="conflict", experiment_id="different")
        self.assertEqual(self.snapshot(), before)

    def test_create_failure_at_last_event_rolls_back_every_table(self):
        before = self.snapshot()
        self.reject_events()
        for create in (self.simulation, self.artifact):
            with (
                self.subTest(create=create.__name__),
                self.assertRaisesRegex(sqlite3.IntegrityError, "event rejected"),
            ):
                create()
            self.assertEqual(self.snapshot(), before)

    def test_reused_experiment_and_relation_rollback_on_event_failure(self):
        self.simulation()
        self.turns.start("c", "next", "again")
        before = self.snapshot()
        self.reject_events()
        with self.assertRaises(sqlite3.IntegrityError):
            self.simulation(job_id="j2", turn_id="next")
        self.assertEqual(self.snapshot(), before)

    def test_update_event_or_callback_failure_rolls_back_status_and_time(self):
        self.simulation()
        before = self.snapshot()
        callback = Mock(side_effect=RuntimeError("callback failed"))
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            self.update(event=callback)
        self.assertEqual(self.snapshot(), before)
        self.reject_events()
        with self.assertRaises(sqlite3.IntegrityError):
            self.update()
        self.assertEqual(self.snapshot(), before)

    def test_wrong_owner_type_or_missing_job_never_calls_event_or_mutates(self):
        self.simulation()
        self.artifact()
        before = self.snapshot()
        callback = Mock()
        for job_id, overrides in (
            ("missing", {}),
            ("j", {"conversation_id": "other"}),
            ("j", {"turn_id": "other-turn"}),
            ("j", {"simulation": False}),
            ("a", {"simulation": True}),
        ):
            with self.subTest(job=job_id, overrides=overrides):
                self.assertIsNone(self.update(job_id, event=callback, **overrides))
                self.assertEqual(self.snapshot(), before)
        callback.assert_not_called()

    def test_create_requires_matching_running_turn_and_terminal_updates_fail(self):
        before = self.snapshot()
        for create in (self.simulation, self.artifact):
            for overrides in ({"turn_id": "missing"}, {"conversation_id": "other"}):
                with self.assertRaises(KeyError):
                    create(**overrides)
                self.assertEqual(self.snapshot(), before)
        self.simulation()
        self.artifact()
        self.turns.finish("c", "c-turn", text="done", status="complete")
        before = self.snapshot()
        for action in (
            lambda: self.simulation(job_id="new"),
            lambda: self.artifact(job_id="new"),
            self.update,
            lambda: self.update("a", simulation=False),
        ):
            with self.assertRaises(ValueError):
                action()
            self.assertEqual(self.snapshot(), before)

    def test_artifact_lookup_projection_and_deterministic_order(self):
        with patch("vibesim_agent.storage.jobs.time.time", return_value=10):
            first = self.artifact(job_id="b", resource_id="rb")
            self.artifact(job_id="a", resource_id="ra")
            self.simulation()
        self.assertEqual(first["artifact_path"], "artifact")
        self.assertEqual(self.jobs.artifact_by_resource("rb")["role"], "assistant")
        self.assertIsNone(self.jobs.artifact_by_resource("absent"))
        self.assertEqual(
            [row["job_id"] for row in self.jobs.list_artifacts()], ["a", "b"]
        )
        with patch("vibesim_agent.storage.jobs.time.time", return_value=20):
            self.update(
                "b",
                simulation=False,
                event=lambda job: {"kind": "job.completed", **job},
            )
        rows = self.jobs.list_artifacts()
        self.assertEqual([row["job_id"] for row in rows], ["b", "a"])
        self.assertEqual(rows[0]["conversation_title"], "c")
        self.assertEqual(rows[0]["updated_at"], 20)
        self.assertNotIn("descriptor", rows[0])
        self.assertNotIn("id", rows[0])
        self.assertEqual(self.turns.events("c", "c-turn")[-1]["kind"], "job.completed")

    def test_experiment_listing_preserves_each_conversation_produced_link(self):
        with patch("vibesim_agent.storage.jobs.time.time", return_value=10):
            self.simulation()
        with patch("vibesim_agent.storage.jobs.time.time", return_value=20):
            self.simulation(
                job_id="other-j", conversation_id="other", turn_id="other-turn"
            )
            self.simulation(job_id="new-j", experiment_id="f", experiment_path="new")
        rows = self.jobs.list_experiments()
        self.assertEqual(
            [(row["id"], row["conversation_id"]) for row in rows],
            [("e", "c"), ("e", "other"), ("f", "c")],
        )
        self.assertEqual(len(self.jobs.list_experiments("other")), 1)
        self.assertEqual(self.jobs.list_experiments("missing"), [])
        self.assertIsNone(self.jobs.get_experiment("missing"))
        self.assertIsNone(self.jobs.experiment_by_path("missing"))

    def test_artifact_cannot_bypass_simulation_identity(self):
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.artifact(job_kind="simulation")
        self.assertEqual(self.snapshot(), before)

    def test_experiment_id_cannot_be_reassigned_to_another_path(self):
        self.simulation()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "different path"):
            self.simulation(job_id="new", experiment_path="other-path")
        self.assertEqual(self.snapshot(), before)

    def test_late_old_job_status_preserves_current_experiment_job(self):
        self.simulation()
        self.turns.start("c", "next", "again")
        self.simulation(job_id="j2", turn_id="next")
        experiment = self.jobs.get_experiment("e")
        updated = self.update()
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(self.jobs.get("j"), {**updated, "role": "assistant"})
        self.assertEqual(self.jobs.get_experiment("e"), experiment)
        self.assertEqual(self.turns.events("c", "c-turn")[-1]["payload"]["job_id"], "j")
        self.update("j2", turn_id="next")
        self.assertEqual(self.jobs.get_experiment("e")["status"], "completed")

    def test_get_returns_owner_and_recovery_identity_without_writing(self):
        self.simulation()
        self.artifact()
        before = self.snapshot()
        self.assertEqual(
            self.jobs.get("j"),
            {
                "job_id": "j",
                "role": "assistant",
                "experiment_id": "e",
                "conversation_id": "c",
                "turn_id": "c-turn",
                "status": "requested",
                "experiment_path": "run",
                "job_kind": "simulation",
                "artifact_path": None,
                "resource_id": None,
                "analyzer_resource_id": None,
            },
        )
        self.assertEqual(self.jobs.get("a")["resource_id"], "r")
        self.assertIsNone(self.jobs.get("missing"))
        self.assertEqual(self.snapshot(), before)
