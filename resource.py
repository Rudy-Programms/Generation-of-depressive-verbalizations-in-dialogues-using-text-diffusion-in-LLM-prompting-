# resource.py — Windows stub for the Unix-only "resource" module that vec2text imports.
import os
import types

RUSAGE_SELF = 0
RLIMIT_AS = 0
RLIMIT_DATA = 1
RLIM_INFINITY = -1

def getrusage(who=RUSAGE_SELF):
    """Return an object with ru_maxrss (KB)."""
    try:
        import psutil  
        rss_kb = psutil.Process(os.getpid()).memory_info().rss // 1024
    except Exception:
        rss_kb = 0
    return types.SimpleNamespace(ru_maxrss=rss_kb)

def getrlimit(resource):
    return (RLIM_INFINITY, RLIM_INFINITY)

def setrlimit(resource, limits):
    # No-op on Windows
    return None
