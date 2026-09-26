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
from compute_fabric.errors import Conflict, Forbidden
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
    """使用文件型 SQLite 与真实线程，确定性覆盖分配的两种竞争结局。"""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "fabric.sqlite3"
        self.connection = connect(self.path)
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = SupplyService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"facility_id": "cluster-a", "name": "北部数据中心", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_gpu_hours": "500000"})
        self.service.create_facility("plan", {"facility_id": "pool-b", "name": "东部推理池", "kind": "inference-pool", "timezone": "Asia/Shanghai", "capacity_gpu_hours": "800000"})
        self.service.create_route("plan", {"route_id": "fabric-a-b", "origin_id": "cluster-a", "destination_id": "pool-b", "product": "gpu-h100", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            self.service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "fabric-a-b", "shipper_id": f"shipper-{number}", "service_date": "2026-09-25", "requested_gpu_hours": requested, "priority": priority, "idempotency_key": f"key-{number}"})

    def tearDown(self) -> None:
        self.connection.close()
        self.directory.cleanup()

    def _thread_service(self) -> SupplyService:
        return SupplyService(connect(self.path), self.clock)

    def test_same_input_race_returns_one_business_result(self) -> None:
        barrier = threading.Barrier(2)
        results: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def worker() -> None:
            connection = connect(self.path)
            try:
                service = SupplyService(connection, self.clock)
                barrier.wait(timeout=10)
                results.append(service.allocate("dispatch", "fabric-a-b", "2026-09-25"))
            except BaseException as exc:  # noqa: BLE001 - 断言中报告任何底层异常
                errors.append(exc)
            finally:
                connection.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(len({result["allocation_id"] for result in results}), 1)
        self.assertEqual(results[0]["allocations"], results[1]["allocations"])
        self.assertEqual(sorted(bool(result["replayed"]) for result in results), [False, True])
        run_count = self.connection.execute(
            "SELECT count(*) FROM allocation_runs WHERE route_id='fabric-a-b' AND service_date='2026-09-25'"
        ).fetchone()[0]
        self.assertEqual(run_count, 1)
        states = self.connection.execute(
            "SELECT state,revision FROM nominations WHERE route_id='fabric-a-b' AND service_date='2026-09-25' "
            "ORDER BY nomination_id"
        ).fetchall()
        self.assertEqual([(row["state"], row["revision"]) for row in states], [("allocated", 2), ("allocated", 2)])

    def test_changed_nomination_set_conflicts_without_partial_state(self) -> None:
        first = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.assertFalse(first["replayed"])

        inserted = threading.Event()

        def submit_extra() -> None:
            connection = connect(self.path)
            try:
                service = SupplyService(connection, self.clock)
                service.submit_nomination("dispatch", {"nomination_id": "nom-3", "route_id": "fabric-a-b", "shipper_id": "shipper-3", "service_date": "2026-09-25", "requested_gpu_hours": "20000", "priority": 30, "idempotency_key": "key-3"})
                inserted.set()
            finally:
                connection.close()

        conflict: list[BaseException] = []

        def retry_allocate() -> None:
            inserted.wait(timeout=10)
            connection = connect(self.path)
            try:
                service = SupplyService(connection, self.clock)
                service.allocate("dispatch", "fabric-a-b", "2026-09-25")
            except BaseException as exc:  # noqa: BLE001
                conflict.append(exc)
            finally:
                connection.close()

        submitter = threading.Thread(target=submit_extra)
        retrying = threading.Thread(target=retry_allocate)
        submitter.start()
        retrying.start()
        submitter.join(timeout=30)
        retrying.join(timeout=30)

        self.assertEqual(len(conflict), 1)
        self.assertIsInstance(conflict[0], Conflict)
        message = str(conflict[0])
        self.assertNotIn("sqlite", message.lower())
        self.assertNotIn("UNIQUE", message)
        extra = self.connection.execute("SELECT * FROM nominations WHERE nomination_id='nom-3'").fetchone()
        self.assertEqual(extra["state"], "submitted")
        self.assertEqual(extra["revision"], 1)
        self.assertEqual(extra["allocated_gpu_hours"], "0")
        run_count = self.connection.execute(
            "SELECT count(*) FROM allocation_runs WHERE route_id='fabric-a-b' AND service_date='2026-09-25'"
        ).fetchone()[0]
        self.assertEqual(run_count, 1)

        app = JsonApplication(self.service)
        response = app.handle(
            "POST",
            "/routes/fabric-a-b/allocate",
            {"X-Actor-Id": "dispatch", "Content-Type": "application/json"},
            b'{"service_date":"2026-09-25"}',
        )
        self.assertEqual(response.status, 409)
        self.assertEqual(response.body["error"]["code"], "conflict")
        self.assertNotIn("sqlite", response.body["error"]["message"].lower())

    def test_maintenance_version_change_turns_replay_into_conflict(self) -> None:
        first = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        repeated = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        self.assertTrue(repeated["replayed"])
        self.assertEqual(repeated["allocation_id"], first["allocation_id"])
        self.service.announce_outage("risk", "fabric-a-b", "2026-09-25T00:00:00Z", "2026-09-25T23:59:59Z", "50", "临时检修")
        with self.assertRaises(Conflict):
            self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")

    def test_audit_endpoint_restores_snapshot_summary_and_commit_version(self) -> None:
        allocation = self.service.allocate("dispatch", "fabric-a-b", "2026-09-25")
        app = JsonApplication(self.service)
        response = app.handle("GET", f"/audit/allocations/{allocation['allocation_id']}", {"X-Actor-Id": "audit"})
        self.assertEqual(response.status, 200)
        body = response.body
        self.assertEqual(body["allocation_id"], allocation["allocation_id"])
        self.assertEqual(body["route_id"], "fabric-a-b")
        self.assertEqual(body["service_date"], "2026-09-25")
        self.assertEqual(body["available_capacity"], "100000.000")
        self.assertEqual(body["capacity_snapshot"]["daily_capacity"], "100000")
        self.assertEqual(body["capacity_snapshot"]["outages"], [])
        self.assertEqual(body["nominations"]["count"], 2)
        self.assertEqual(body["nominations"]["ids"], ["nom-1", "nom-2"])
        self.assertEqual(len(body["nominations"]["sha256"]), 64)
        self.assertEqual(len(body["context_sha256"]), 64)
        self.assertEqual(len(body["input_sha256"]), 64)
        self.assertEqual(body["allocations"], allocation["allocations"])
        stored = self.connection.execute(
            "SELECT input_sha256 FROM allocation_runs WHERE allocation_id=?",
            (allocation["allocation_id"],),
        ).fetchone()
        self.assertEqual(body["input_sha256"], stored["input_sha256"])

        forbidden = app.handle("GET", f"/audit/allocations/{allocation['allocation_id']}", {"X-Actor-Id": "dispatch"})
        self.assertEqual(forbidden.status, 403)
        missing = app.handle("GET", "/audit/allocations/9999", {"X-Actor-Id": "audit"})
        self.assertEqual(missing.status, 404)


if __name__ == "__main__":
    unittest.main()
