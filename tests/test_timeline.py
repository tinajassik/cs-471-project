"""Checks the trace summary on small databases with the tables of an nsys export."""
import sqlite3

import pytest

from analysis.analyse_timeline import PROCESS_SHIFT, busy_time, phase_stats, summarise

PROCESS_A = 100
PROCESS_B = 200


def thread_of(process):
    return process * PROCESS_SHIFT + 1


def make_database(path, phases, kernels):
    """phases holds (process, start, end, name); kernels holds (process, correlation, launch, start, end)."""
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INT, "end" INT, eventType INT, text TEXT, globalTid INT, textId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, "end" INT, globalTid INT, correlationId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, "end" INT, globalPid INT, correlationId INT);
    """)
    for process, start, end, name in phases:
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 59, ?, ?, NULL)", (start, end, name, thread_of(process)))
    for process, correlation, launch, start, end in kernels:
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                           (launch, launch + 1, thread_of(process), correlation))
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
                           (start, end, process * PROCESS_SHIFT, correlation))
    connection.commit()
    connection.close()


def test_busy_time_counts_overlaps_once():
    assert busy_time([(0, 10), (5, 15), (20, 30), (22, 25)]) == 25


def test_cpu_limited_phase_shows_an_idle_gpu():
    # Launches every 100 us and each kernel takes 10 us.
    kernels = [(i * 100_000, i * 100_000 + 1_000, i * 100_000 + 11_000) for i in range(10)]
    stats = phase_stats(0, 1_000_000, kernels)
    assert stats["kernels"] == 10
    assert stats["launch_gap_us"] == pytest.approx(100)
    assert stats["kernel_us"] == pytest.approx(10)
    assert stats["busy_us"] == pytest.approx(100)
    assert stats["span_us"] == pytest.approx(910)
    assert stats["gpu_after_cpu_us"] < 0


def test_gpu_limited_phase_shows_a_busy_gpu_finishing_late():
    # Launches every 10 us, but each kernel takes 100 us, so they queue up.
    kernels = [(i * 10_000, i * 100_000, (i + 1) * 100_000) for i in range(10)]
    stats = phase_stats(0, 100_000, kernels)
    assert stats["busy_us"] == pytest.approx(stats["span_us"])
    assert stats["gpu_after_cpu_us"] == pytest.approx(900)


def test_phase_uses_launch_time_rather_than_when_the_kernel_ran():
    late_kernel = (50, 500, 600)
    other_phase_kernel = (150, 160, 170)
    stats = phase_stats(0, 100, [late_kernel, other_phase_kernel])
    assert stats["kernels"] == 1
    assert phase_stats(200, 300, [late_kernel, other_phase_kernel]) is None


def test_summary_keeps_processes_apart_with_repeated_correlation_ids(tmp_path):
    path = tmp_path / "trace.sqlite"
    make_database(
        path,
        [(PROCESS_A, 0, 1000, "compute"), (PROCESS_B, 0, 1000, "compute")],
        [(PROCESS_A, 1, 10, 20, 120), (PROCESS_B, 1, 10, 20, 320)],
    )
    summaries = summarise(path)
    assert summaries[PROCESS_A]["busy_us"] == pytest.approx(0.1)
    assert summaries[PROCESS_B]["busy_us"] == pytest.approx(0.3)


def test_summary_ignores_other_phases_and_earlier_steps(tmp_path):
    path = tmp_path / "trace.sqlite"
    make_database(
        path,
        [(PROCESS_A, 0, 1000, "dispatch"), (PROCESS_A, 2000, 3000, "compute"), (PROCESS_A, 4000, 5000, "compute")],
        [(PROCESS_A, 1, 10, 20, 9020), (PROCESS_A, 2, 2010, 2020, 7020), (PROCESS_A, 3, 4010, 4020, 4120)],
    )
    summary = summarise(path, last=1)[PROCESS_A]
    assert summary["phases"] == 1
    assert summary["busy_us"] == pytest.approx(0.1)


def test_missing_file_fails_without_creating_an_empty_database(tmp_path):
    path = tmp_path / "timeline_*_1.sqlite"
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        summarise(path)
    assert not path.exists()


def test_summary_reads_a_phase_name_stored_as_a_string_id(tmp_path):
    path = tmp_path / "trace.sqlite"
    make_database(path, [], [(PROCESS_A, 1, 10, 20, 120)])
    connection = sqlite3.connect(path)
    connection.execute("INSERT INTO StringIds VALUES (7, 'compute')")
    connection.execute("INSERT INTO NVTX_EVENTS VALUES (0, 1000, 59, NULL, ?, 7)", (thread_of(PROCESS_A),))
    connection.commit()
    connection.close()
    assert summarise(path)[PROCESS_A]["kernels"] == 1
