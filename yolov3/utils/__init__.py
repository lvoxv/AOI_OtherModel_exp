# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""utils/initialization."""

import contextlib
import platform
import threading


def emojis(str=""):
    """Returns platform-dependent emoji-safe version of str; ignores emojis on Windows, else returns original str."""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str


class TryExcept(contextlib.ContextDecorator):
    """A context manager and decorator for handling exceptions with optional custom messages."""

    def __init__(self, msg=""):
        """Initializes TryExcept with optional custom message, used as decorator or context manager for exception
        handling.
        """
        self.msg = msg

    def __enter__(self):
        """Begin exception-handling block, optionally customizing exception message when used with TryExcept context
        manager.
        """
        pass

    def __exit__(self, exc_type, value, traceback):
        """Ends exception-handling block, optionally prints custom message with exception, suppressing exceptions within
        context.
        """
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """
    Decorates a function to run in a separate thread, returning the thread object.

    Usage: @threaded.
    """

    def wrapper(*args, **kwargs):
        """
        Runs the decorated function in a separate thread and returns the thread object.

        Usage: @threaded.
        """
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    """Joins all daemon threads, excluding the main thread, with an optional verbose flag for logging."""
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f"Joining thread {t.name}")
            t.join()


def notebook_init(verbose=True):
    """Initializes notebook environment by checking hardware, software requirements, and cleaning up if in Colab."""
    print("Checking setup...")

    import os
    import shutil

    from ultralytics.utils.checks import check_requirements

    from utils.general import check_font, is_colab
    from utils.torch_utils import select_device  # imports

    check_font()

    import psutil

    if check_requirements("wandb", install=False):
        os.system("pip uninstall -y wandb")  # eliminate unexpected account creation prompt with infinite hang
    if is_colab():
        shutil.rmtree("/content/sample_data", ignore_errors=True)  # remove colab /sample_data directory

    # System info
    display = None
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display

            display.clear_output()
        s = f"({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)"
    else:
        s = ""

    select_device(newline=False)
    print(emojis(f"Setup complete ✅ {s}"))
    return display
