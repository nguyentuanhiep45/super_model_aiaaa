import psutil as ps

print(ps.Process(ps.pids()[0]))