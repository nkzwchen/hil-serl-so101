"""One processing job and one replaceable pending frame per camera."""
import threading


class LatestFrameWorker:
    def __init__(self, callback, name, on_error):
        self._callback = callback
        self._on_error = on_error
        self._condition = threading.Condition()
        self._pending = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, message, arrival):
        with self._condition:
            if self._stopped:
                return False
            replaced = self._pending is not None
            self._pending = (message, arrival)
            self._condition.notify()
            return replaced

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopped or self._pending is not None)
                if self._stopped:
                    return
                job, self._pending = self._pending, None
            try:
                self._callback(*job)
            except Exception as exc:
                self._on_error(exc)

    def close(self):
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=2.)
