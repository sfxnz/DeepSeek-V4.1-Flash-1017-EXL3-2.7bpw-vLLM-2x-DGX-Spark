"""Docker's --restart on-failure:3 as the gateway uses it (serve image, sitecustomize masked, CPU only).

A: a process that is OOM-killed by its cgroup (exit 137) is restarted, at most 3 times, then stays exited.
   Polls `docker ps` (running list, what run.sh's wait checks) and `docker inspect` while it loops.
B: `docker rm -f` during the restart loop removes it for good (stop.sh, the EXIT trap, run.sh's leftover
   handling all use it): nothing comes back.
"""
import subprocess
import time

IMAGE = "dsv41-flash-exl3-sm121:e16-rows"
NAME = "dsv41-restart-policy-check"
BASE = ["docker", "run", "-d", "--name", NAME, "--restart", "on-failure:3", "--network", "none",
        "--memory", "256m", "--memory-swap", "256m",
        "-v", "/dev/null:/usr/lib/python3.12/sitecustomize.py:ro", "--entrypoint", "python3", IMAGE, "-c"]
# Grows past the 256m cgroup 1 s after start: the kernel OOM-kills it (as an image burst past 3g would).
OOM = "import time; time.sleep(1); b = bytearray(1 << 30); time.sleep(3600)"


def sh(*args):
    return subprocess.run(list(args), capture_output=True, text=True).stdout.strip()


def state():
    return sh("docker", "inspect", "-f", "{{.RestartCount}} {{.State.Status}} oom={{.State.OOMKilled}} "
              "exit={{.State.ExitCode}} restarting={{.State.Restarting}}", NAME)


def in_ps():
    return NAME in sh("docker", "ps", "--format", "{{.Names}}").split()


sh("docker", "rm", "-f", NAME)
print("A: OOM kill under on-failure:3")
subprocess.run(BASE + [OOM], check=True, capture_output=True)
t0, seen, last = time.time(), [], None
while time.time() - t0 < 40:
    s = f"{state()} in_ps={in_ps()}"
    if s != last:
        print(f"  t={time.time() - t0:5.1f}s {s}")
        seen.append(s)
        last = s
    if " exited " in s and s.startswith("3 "):
        break
    time.sleep(0.2)
time.sleep(5)
print(f"  5 s later: {state()} in_ps={in_ps()}")
sh("docker", "rm", "-f", NAME)

print("B: docker rm -f while it restarts")
subprocess.run(BASE + [OOM], check=True, capture_output=True)
time.sleep(1.8)
print(f"  before rm: {state()} in_ps={in_ps()}")
sh("docker", "rm", "-f", NAME)
time.sleep(8)
left = sh("docker", "ps", "-a", "--filter", f"name=^{NAME}$", "--format", "{{.Names}} {{.Status}}")
print(f"  8 s after rm -f: {left or 'no container'}")
