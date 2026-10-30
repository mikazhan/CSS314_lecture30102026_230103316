import subprocess
import platform
import sys
import json
import argparse
from salt import get_student_params

physical = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
logical = subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True).strip()
cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
l1 = subprocess.check_output(["sysctl", "-n", "hw.l1dcachesize"], text=True).strip()
l2 = subprocess.check_output(["sysctl", "-n", "hw.l2cachesize"], text=True).strip()
cache_line = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], text=True).strip()

profile = {
    "cpu": cpu,
    "physical_cores": int(physical),
    "logical_cores": int(logical),
    "l1_cache": int(l1),
    "l2_cache": int(l2),
    "l3_cache": None,
    "cache_line": int(cache_line),
    "os": platform.platform(),
    "kernel": platform.release(),
    "architecture": platform.machine(),
    "python": sys.version
}

with open("results/host_profile.json", "w") as f:
    json.dump(profile, f, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("--student-id", required=True)
args = parser.parse_args()

params = get_student_params(args.student_id)

with open("results/salted_params.json", "w") as f:
    json.dump(params, f, indent=4)

print(profile)
print(params)