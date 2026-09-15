"""Versioned, backend-independent Phase 4 result protocol.

The result protocol deliberately has no dependency on Inspect, Pi, SGLang, or
the dataset loader.  A runner can therefore adapt its native sample object to
these plain JSON records and use the same aggregation and integrity checks for
every backend.

The public helpers in this module are intentionally small and stdlib-only.  A
validator returns ``True`` on success and raises :class:`ResultValidationError`
on malformed or unsafe records.  This makes validation useful both as an
assertion in a runner and as a fail-closed check before a run is marked
complete.
"""

from .protocol import *  # noqa: F401,F403
from .protocol import __all__

