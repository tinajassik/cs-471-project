from moe_bench.numa_bind import parse_cpu_list, sysfs_device


def test_cpu_list_with_ranges_and_single_cpus():
    assert parse_cpu_list("0-2,8,10-11\n") == {0, 1, 2, 8, 10, 11}


def test_cpu_list_with_one_cpu():
    assert parse_cpu_list("5") == {5}


def test_bus_id_matches_the_sysfs_name():
    assert sysfs_device("00000000:1B:00.0\n") == "0000:1b:00.0"
