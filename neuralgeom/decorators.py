import time


def timer(func):
    """Decorator function to time the execution of a function."""
    
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func