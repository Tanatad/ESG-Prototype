from memory_profiler import profile
from app.main import start

@profile
def profiled_start():
    start()

if __name__ == "__main__":
    profiled_start()
