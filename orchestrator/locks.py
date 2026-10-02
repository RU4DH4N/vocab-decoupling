import fcntl
import hashlib
import os
import tempfile
from pathlib import Path


class Lock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError(f"already in use: {self.path}") from None
        return self

    def __exit__(self, *_: object) -> None:
        self.handle.close()


def resource_lock(resource: str) -> Lock:
    name = hashlib.sha256(resource.encode()).hexdigest()
    return Lock(
        Path(tempfile.gettempdir()) / f"vdr-resources-{os.getuid()}" / f"{name}.lock"
    )
