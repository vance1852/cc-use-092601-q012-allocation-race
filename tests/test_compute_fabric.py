from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from compute_fabric.api import JsonApplication
from compute_fabric.clock import FrozenClock
from compute_fabric.errors import Conflict, Forbidden, InvalidState, NotFound
from compute_fabric.planning import AllocationRequest, PricePoint, allocate_capacity, latest_streak
from compute_fabric.service import SupplyService
from compute_fabric.storage import connect
from compute_fabric.risk import DemandBucket, inventory_coverage, mark_to_market, supply_gap


class PlanningTests(unittest.TestCase):
    def test_latest_down_streak_uses_first_close_as_base(self) -> None:
        streak = latest_streak([
            PricePoint("2026-09-18", Decimal("108")),
            PricePoint("2026-09-19", Decimal("105")),
            PricePoint("2026-09-20", Decimal("102")),
            PricePoint("2026-09-21", Decimal("98")),
        ])
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 4)
        self.assertEqual(streak.start_date, "2026-09-18")
        self.assertEqual(streak.end_close, Decimal("98"))

    def test_allocation_is_stable_and_does_not_exceed_capacity(self) -> None:
        rows = allocate_capacity(Decimal("100"), [
            AllocationRequest("later", Decimal("80"), 20, "2026-09-24T09:00:00Z"),
            AllocationRequest("first", Decimal("70"), 10, "2026-09-24T10:00:00Z"),
        ])
        self.assertEqual(rows[0]["nomination_id"], "first")
        self.assertEqual(rows[0]["allocated_gpu_hours"], "70.000")
        self.assertEqual(rows[1]["allocated_gpu_hours"], "30.000")

    def test_inventory_coverage_and_supply_gap(self) -> None:
        coverage = inventory_coverage(
            [{"facility_id": "inference-pool", "product": "gpu-a100", "available_gpu_hours": "250"}],
            [DemandBucket("inference-pool", "gpu-a100", Decimal("100"), Decimal("20"))],
        )
        self.assertEqual(coverage[0]["coverage_days"], "2.30")
        self.assertTrue(coverage[0]["below_three_days"])
        gap = supply_gap(
            opening_inventory=Decimal("100"),
            confirmed_inbound=Decimal("30"),
            forecast_demand=Decimal("120"),
            protected_reserve=Decimal("40"),
        )
        self.assertEqual(gap["supply_gap"], "30.000")

    def test_mark_to_market_groups_deterministically(self) -> None:
        result = mark_to_market(
            [{"position_id": "p1", "market_index": "PEAK_VALLEY", "quantity_gpu_hours": "100", "entry_price_cny": "105"}],
            {"PEAK_VALLEY": Decimal("98")},
        )
        self.assertEqual(result["unrealized_pnl_cny"], "-700.00")


class SupplyServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = SupplyService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"facility_id": "cluster-a", "name": "北部数据中心", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_gpu_hours": "500000"})
        self.service.create_facility("plan", {"facility_id": "pool-b", "name": "东部推理池", "kind": "inference-pool", "timezone": "Asia/Shanghai", "capacity_gpu_hours": "800000"})
        self.service.create_route("plan", {"route_id": "fabric-a-b", "origin_id": "cluster-a", "destination_id": "pool-b", "product": "gpu-h100", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})

    def tearDown(self) -> None:
        self.connection.close()

    def quote(self, day: int, close: str) -> dict[str, object]:
        return self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": f"2026-09-{day}", "close_cny": close, "source_revision": f"r-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})

    def test_quote_revisions_preserve_history(self) -> None:
        first = self.quote(23, "98")
        second = self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": "2026-09-23", "close_cny": "97.8", "source_revision": "r-23-corrected", "observed_at": "2026-09-23T22:00:00Z"})
        self.assertNotEqual(first["quote_id"], second["quote_id"])
        rows = self.connection.execute("SELECT * FROM market_index_quotes ORDER BY quote_id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["supersedes_quote_id"], rows[0]["quote_id"])

    def test_nomination_replay_and_payload_conflict(self) -> None:
        payload = {"nomination_id": "nom-1", "route_id": "fabric-a-b", "shipper_id": "tenant", "service_date": "2026-09-25", "requested_gpu_hours": "80000", "priority": 10, "idempotency_key": "key-1"}
        first = self.service.submit_nomination("dispatch", payload)
        self.assertEqual(first, self.service.submit_nomination("dispatch", payload))
        changed = dict(payload, requested_gpu_hours="81000")
        with self.assertRaises(Conflict):
            self.service.submit_nomination("dispatch", changed)

    def test_outage_reduces_allocation_and_transfer_consumes_inventory(self) -> None:
        self.service.announce_outage("risk", "fabric-a-b", "2026-09-25T00:00:00Z", "2026-09-25T23:59:59Z", "50", "检修")
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            self.service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "fabric-a-b", "shipper_id": f"shipper-{number}", "service_date": "2026-09-25", "requested_gpu_hours": requested, "priority": priority, "idempotency_key": f"key-{number}"})
        allocation = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.assertEqual(allocation["available_capacity"], "50000.000")
        self.assertEqual(allocation["allocations"][1]["allocated_gpu_hours"], "10000.000")
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "cluster-a", "product": "gpu-h100", "grade": "PEAK_VALLEY", "quantity_gpu_hours": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        transfer = self.service.dispatch_transfer("dispatch", "transfer-1", "nom-1", "lot-1", 2)
        self.assertEqual(transfer["loaded_gpu_hours"], "40000.000")
        self.assertEqual(self.service.inventory_lot("lot-1")["available_gpu_hours"], "20000.000")

    def nominate(self, number: int, requested: str, priority: int, service_date: str = "2026-09-25") -> dict[str, object]:
        return self.service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "fabric-a-b", "shipper_id": f"shipper-{number}", "service_date": service_date, "requested_gpu_hours": requested, "priority": priority, "idempotency_key": f"key-{number}"})

    def test_allocate_retry_replays_committed_result(self) -> None:
        self.nominate(1, "80000", 10)
        self.nominate(2, "60000", 20)
        first = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.assertFalse(first["replayed"])
        second = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.assertTrue(second["replayed"])
        self.assertEqual(first["allocation_id"], second["allocation_id"])
        self.assertEqual(first["allocations"], second["allocations"])
        self.assertEqual(first["available_capacity"], second["available_capacity"])
        runs = self.connection.execute("SELECT * FROM allocation_runs").fetchall()
        self.assertEqual(len(runs), 1)

    def test_allocate_conflicts_when_nomination_set_changed(self) -> None:
        self.nominate(1, "80000", 10)
        self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.nominate(2, "10000", 5)
        with self.assertRaises(Conflict):
            self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        pending = self.connection.execute("SELECT state,revision FROM nominations WHERE nomination_id='nom-2'").fetchone()
        self.assertEqual((pending["state"], pending["revision"]), ("submitted", 1))
        runs = self.connection.execute("SELECT * FROM allocation_runs").fetchall()
        self.assertEqual(len(runs), 1)

    def test_allocate_replays_committed_result_despite_later_maintenance(self) -> None:
        self.nominate(1, "80000", 10)
        first = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.service.announce_outage("risk", "fabric-a-b", "2026-09-25T00:00:00Z", "2026-09-25T23:59:59Z", "50", "检修")
        replay = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["available_capacity"], first["available_capacity"])
        self.nominate(2, "10000", 5)
        with self.assertRaises(Conflict):
            self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")

    def test_allocate_without_pending_nominations_still_invalid(self) -> None:
        with self.assertRaises(InvalidState):
            self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        with self.assertRaises(NotFound):
            self.service.allocate("dispatch", "fabric-missing", "2026-09-25")

    def test_failed_allocation_leaves_no_partial_nomination_state(self) -> None:
        self.nominate(1, "80000", 10)
        self.nominate(2, "60000", 20)
        original_audit = self.service._audit

        def failing_audit(entity_type, entity_id, event_type, actor_id, payload):  # noqa: ANN001, ANN202
            if event_type == "allocation.completed":
                raise RuntimeError("模拟审计写入失败")
            return original_audit(entity_type, entity_id, event_type, actor_id, payload)

        self.service._audit = failing_audit
        with self.assertRaises(RuntimeError):
            self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        rows = self.connection.execute("SELECT state,revision,allocated_gpu_hours FROM nominations ORDER BY nomination_id").fetchall()
        self.assertEqual([(row["state"], row["revision"], row["allocated_gpu_hours"]) for row in rows], [("submitted", 1, "0"), ("submitted", 1, "0")])
        runs = self.connection.execute("SELECT * FROM allocation_runs").fetchall()
        self.assertEqual(runs, [])
        self.service._audit = original_audit
        recovered = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.assertFalse(recovered["replayed"])

    def test_allocation_audit_restores_snapshot_digest_and_commit_version(self) -> None:
        self.service.announce_outage("risk", "fabric-a-b", "2026-09-25T00:00:00Z", "2026-09-25T12:00:00Z", "50", "检修")
        self.nominate(1, "40000", 10)
        self.nominate(2, "30000", 20)
        allocation = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        audit = self.service.allocation_audit("audit", "fabric-a-b", "2026-09-25")
        self.assertEqual(audit["allocation_id"], allocation["allocation_id"])
        self.assertEqual(audit["available_capacity"], "50000.000")
        snapshot = audit["capacity_snapshot"]
        self.assertEqual(snapshot["route_id"], "fabric-a-b")
        self.assertEqual(snapshot["route_revision"], 1)
        self.assertEqual(snapshot["daily_capacity"], "100000")
        self.assertEqual(snapshot["available_capacity"], "50000.000")
        self.assertEqual(len(snapshot["outages"]), 1)
        self.assertEqual(snapshot["outages"][0]["capacity_percent"], "50")
        self.assertEqual(snapshot["outages"][0]["revision"], 1)
        self.assertEqual(len(audit["input_sha256"]), 64)
        self.assertEqual(len(audit["nominations_sha256"]), 64)
        self.assertNotEqual(audit["input_sha256"], audit["nominations_sha256"])
        self.assertEqual(audit["committed_by"], "dispatch")
        self.assertEqual(audit["committed_at"], "2026-09-24T08:00:00Z")
        self.assertEqual(audit["result"]["allocations"], allocation["allocations"])
        with self.assertRaises(Forbidden):
            self.service.allocation_audit("dispatch", "fabric-a-b", "2026-09-25")
        with self.assertRaises(NotFound):
            self.service.allocation_audit("audit", "fabric-a-b", "2026-09-26")

    def test_api_allocate_replay_conflict_and_audit_endpoint(self) -> None:
        app = JsonApplication(self.service)
        self.nominate(1, "80000", 10)
        body = json.dumps({"service_date": "2026-09-25"}).encode("utf-8")
        first = app.handle("POST", "/routes/fabric-a-b/allocate", {"X-Actor-Id": "dispatch"}, body)
        self.assertEqual(first.status, 200)
        self.assertFalse(first.body["replayed"])
        replay = app.handle("POST", "/routes/fabric-a-b/allocate", {"X-Actor-Id": "dispatch"}, body)
        self.assertEqual(replay.status, 200)
        self.assertTrue(replay.body["replayed"])
        self.assertEqual(replay.body["allocation_id"], first.body["allocation_id"])
        audit = app.handle("GET", "/routes/fabric-a-b/allocations?service_date=2026-09-25", {"X-Actor-Id": "audit"})
        self.assertEqual(audit.status, 200)
        self.assertEqual(audit.body["allocation_id"], first.body["allocation_id"])
        self.assertEqual(audit.body["capacity_snapshot"]["daily_capacity"], "100000")
        forbidden = app.handle("GET", "/routes/fabric-a-b/allocations?service_date=2026-09-25", {"X-Actor-Id": "dispatch"})
        self.assertEqual(forbidden.status, 403)
        missing = app.handle("GET", "/routes/fabric-a-b/allocations", {"X-Actor-Id": "audit"})
        self.assertEqual(missing.status, 422)
        self.nominate(2, "10000", 5)
        conflict = app.handle("POST", "/routes/fabric-a-b/allocate", {"X-Actor-Id": "dispatch"}, body)
        self.assertEqual(conflict.status, 409)
        self.assertEqual(conflict.body["error"]["code"], "conflict")
        self.assertNotIn("UNIQUE", json.dumps(conflict.body, ensure_ascii=False))
        self.assertNotIn("sqlite", json.dumps(conflict.body, ensure_ascii=False).lower())

    def test_api_does_not_leak_storage_exceptions(self) -> None:
        app = JsonApplication(self.service)
        self.connection.close()
        response = app.handle("GET", "/quotes/summary/PEAK_VALLEY", {"X-Actor-Id": "plan"})
        self.assertEqual(response.status, 500)
        self.assertEqual(response.body["error"]["code"], "storage_error")
        message = response.body["error"]["message"].lower()
        self.assertNotIn("sqlite", message)
        self.assertNotIn("closed", message)

    def test_scenario_is_approved_and_replayed_by_input(self) -> None:
        self.quote(23, "98")
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "cluster-a", "product": "gpu-h100", "grade": "PEAK_VALLEY", "quantity_gpu_hours": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        self.service.create_scenario("plan", {"scenario_id": "restart", "name": "机组检修恢复", "market_index_drop_percent": "9", "route_capacity_changes": {"fabric-a-b": "20"}, "demand_changes": {"cluster-a:gpu-h100": "-5"}})
        with self.assertRaises(Forbidden):
            self.service.approve_scenario("plan", "restart", 1)
        self.service.approve_scenario("risk", "restart", 1)
        first = self.service.run_scenario("plan", "restart", "2026-09-23")
        second = self.service.run_scenario("plan", "restart", "2026-09-23")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["run_id"], second["run_id"])

    def test_audit_chain_detects_tampering(self) -> None:
        self.assertTrue(self.service.audit_chain("audit")["valid"])
        self.connection.execute("UPDATE supply_audit_events SET payload_json='{}' WHERE event_id=1")
        self.assertFalse(self.service.audit_chain("audit")["valid"])

    def test_api_exposes_browser_free_boundary(self) -> None:
        app = JsonApplication(self.service)
        self.assertEqual(app.handle("GET", "/health").status, 200)
        response = app.handle("GET", "/quotes/summary/PEAK_VALLEY", {"X-Actor-Id": "plan"})
        self.assertEqual(response.status, 404)
        self.assertEqual(response.body["error"]["code"], "not_found")


class AllocationConcurrencyTests(unittest.TestCase):
    """两个调度请求竞争同一通道和服务日的确定性结局。"""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db_path = Path(self.tempdir.name) / "fabric.sqlite3"
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.connection = connect(self.db_path)
        self.addCleanup(self.connection.close)
        self.service = SupplyService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"facility_id": "cluster-a", "name": "北部数据中心", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_gpu_hours": "500000"})
        self.service.create_facility("plan", {"facility_id": "pool-b", "name": "东部推理池", "kind": "inference-pool", "timezone": "Asia/Shanghai", "capacity_gpu_hours": "800000"})
        self.service.create_route("plan", {"route_id": "fabric-a-b", "origin_id": "cluster-a", "destination_id": "pool-b", "product": "gpu-h100", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})
        for number, requested, priority in ((1, "80000", 10), (2, "60000", 20), (3, "50000", 30)):
            self.service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "fabric-a-b", "shipper_id": f"shipper-{number}", "service_date": "2026-09-25", "requested_gpu_hours": requested, "priority": priority, "idempotency_key": f"key-{number}"})

    def allocate_concurrently(self, workers: int = 2) -> tuple[list[dict[str, object]], list[BaseException]]:
        barrier = threading.Barrier(workers)
        results: list[dict[str, object]] = []
        failures: list[BaseException] = []
        lock = threading.Lock()

        def worker() -> None:
            connection = connect(self.db_path)
            try:
                service = SupplyService(connection, self.clock)
                barrier.wait(timeout=10)
                outcome = service.allocate("dispatch", "fabric-a-b", "2026-09-25")
                with lock:
                    results.append(outcome)
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    failures.append(exc)
            finally:
                connection.close()

        threads = [threading.Thread(target=worker, name=f"allocator-{index}") for index in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive(), "分配线程未在限期内完成")
        return results, failures

    def test_concurrent_same_input_commits_one_run_and_replays(self) -> None:
        results, failures = self.allocate_concurrently()
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(len({item["allocation_id"] for item in results}), 1)
        self.assertEqual(sorted(item["replayed"] for item in results), [False, True])
        comparable = [{key: value for key, value in item.items() if key != "replayed"} for item in results]
        self.assertEqual(comparable[0], comparable[1])
        runs = self.connection.execute("SELECT * FROM allocation_runs").fetchall()
        self.assertEqual(len(runs), 1)
        nominations = self.connection.execute(
            "SELECT nomination_id,state,revision,allocated_gpu_hours FROM nominations ORDER BY nomination_id"
        ).fetchall()
        self.assertEqual(
            [(row["state"], row["revision"]) for row in nominations],
            [("allocated", 2), ("allocated", 2), ("cancelled", 2)],
        )
        self.assertEqual([row["allocated_gpu_hours"] for row in nominations], ["80000.000", "20000.000", "0.000"])
        self.assertTrue(self.service.audit_chain("audit")["valid"])

    def test_concurrent_changed_input_set_conflicts_without_partial_state(self) -> None:
        first = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.assertFalse(first["replayed"])
        self.service.submit_nomination("dispatch", {"nomination_id": "nom-4", "route_id": "fabric-a-b", "shipper_id": "shipper-4", "service_date": "2026-09-25", "requested_gpu_hours": "10000", "priority": 5, "idempotency_key": "key-4"})
        results, failures = self.allocate_concurrently()
        self.assertEqual(results, [])
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(isinstance(exc, Conflict) for exc in failures))
        runs = self.connection.execute("SELECT * FROM allocation_runs").fetchall()
        self.assertEqual(len(runs), 1)
        pending = self.connection.execute("SELECT state,revision FROM nominations WHERE nomination_id='nom-4'").fetchone()
        self.assertEqual((pending["state"], pending["revision"]), ("submitted", 1))
        committed = self.connection.execute("SELECT state,revision,allocated_gpu_hours FROM nominations WHERE nomination_id='nom-1'").fetchone()
        self.assertEqual((committed["state"], committed["revision"], committed["allocated_gpu_hours"]), ("allocated", 2, "80000.000"))
        self.assertTrue(self.service.audit_chain("audit")["valid"])


if __name__ == "__main__":
    unittest.main()
