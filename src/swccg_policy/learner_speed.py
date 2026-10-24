"""Plan learner-speed: learner speed options (all off by default).

S-2 compiles the embedding-table builds: `fixed_blocks` projects the card and template tables in 64-row blocks, so
every training forward and backward launches thousands of small kernels (93% of the Linear calls of a minibatch). The
tables depend only on parameters, so their shapes are static: one graph per table, grad mode and precision. The
compiled functions live in a registry outside the modules, so copies (export_ts, update_traced) stay eager.
S-4 decodes the next batch's trajectory files while the current update runs.
S-5 (added with the user's approval, 2026-10-09) builds each table with one call over all blocks: the same row-wise
math in one GEMM per layer. Like S-2 it is registered per module, so copies and exports keep the fixed blocks.
"""

import contextlib
import json
import queue
import threading
import time
import weakref

import torch

_COMPILED = weakref.WeakKeyDictionary()
_PAUSED = [False]


_LOCAL = threading.local()


def compiled_table(module):
    """The compiled table build registered for this module (a callable), or None (unregistered, paused, inside a
    trace / compile, or already inside this thread's compiled call: the eager body runs there)."""
    if (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting() or _PAUSED[0]
            or getattr(_LOCAL, 'inside', False)):
        return None
    function = _COMPILED.get(module)
    return None if function is None else lambda: _guarded(function)


def _guarded(function):
    _LOCAL.inside = True
    try:
        return function()
    finally:
        _LOCAL.inside = False


@contextlib.contextmanager
def eager_tables():
    """The eager table builds while inside (parity baselines); the compiled graphs are kept."""
    saved, _PAUSED[0] = _PAUSED[0], True
    try:
        yield
    finally:
        _PAUSED[0] = saved


_SINGLE = weakref.WeakSet()


def single_gemm(module):
    """Plan learner-speed S-5: True if this module builds its table in one call (registered by the learner only)."""
    return not _PAUSED[0] and module in _SINGLE


def single_tables(*models, on=True):
    for model in models:
        for module in table_modules(model):
            (_SINGLE.add if on else _SINGLE.discard)(module)


def table_modules(model):
    """The table-building modules of an EfficientPolicy (actor) or EfficientCritic."""
    encoder = model if hasattr(model, 'cards') else model.encoder  # the actor's .encoder is its transformer
    names = ('cards', 'context', 'qtmpl') + (('otmpl',) if encoder.actor else ())
    return [getattr(encoder, name) for name in names]


def compile_tables(*models):
    for model in models:
        for module in table_modules(model):
            _COMPILED[module] = torch.compile(type(module).table.__get__(module), dynamic=False)


def uncompile_tables(*models):
    for model in models:
        for module in table_modules(model):
            _COMPILED.pop(module, None)


class Prefetch:
    """Plan learner-speed S-4: a thread that collects the next batch's files (in the learner's order) and decodes them
    (`load(files)` -> games) while the update runs. One batch ahead at most. The learner moves the files out of the
    queue after taking them; claimed files are never offered twice."""

    def __init__(self, folder, batch_steps, files_of, load, stopped):
        self.folder, self.batch_steps, self.files_of, self.load, self.stopped = folder, batch_steps, files_of, load, stopped
        self.ready = queue.Queue(maxsize=1)
        self.claimed = set()
        self.error = None
        self.thread = threading.Thread(target=self.run, name='learner-prefetch', daemon=True)
        self.thread.start()

    def collect(self):
        files, count = [], 0
        while count < self.batch_steps:
            if self.stopped():
                return None, 0
            for path in self.files_of(self.folder):
                if path in self.claimed:
                    continue
                sidecar = path.with_name(path.name + '.meta.json')
                if not sidecar.exists():
                    continue
                count += int(json.loads(sidecar.read_text())['steps'])
                files.append(path)
                self.claimed.add(path)
                if count >= self.batch_steps:
                    break
            if count < self.batch_steps:
                time.sleep(1)
        return files, count

    def run(self):
        try:
            while not self.stopped():
                waited = time.monotonic()
                files, _ = self.collect()
                if files is None:
                    break
                ready = time.monotonic()
                games = self.load(files)
                item = {'files': files, 'games': games, 'collect_seconds': ready - waited,
                        'decode_seconds': time.monotonic() - ready}
                while not self.stopped():
                    try:
                        self.ready.put(item, timeout=1)
                        break
                    except queue.Full:
                        continue
        except BaseException as error:  # noqa: BLE001 - handed to the learner thread
            self.error = error

    def get(self):
        """The next decoded batch, or None once stopped (the learner checks its own deadline / STOP)."""
        while True:
            if self.error is not None:
                raise RuntimeError('prefetch thread failed') from self.error
            try:
                return self.ready.get(timeout=1)
            except queue.Empty:
                if self.stopped() or not self.thread.is_alive():
                    if self.error is not None:
                        raise RuntimeError('prefetch thread failed') from self.error
                    return None
