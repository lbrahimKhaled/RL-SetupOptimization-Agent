"""Per-platform golden values.

Rollout digests and cache keys hash float bits and the ocd_native binary, so they are only
reproducible on the machine type that recorded them (different compiler, libm and CPU give
different last bits). Each golden is a dict platform -> value; `golden(d)` returns this platform's
value, or skips the test where none was recorded.

  mac      macOS arm64 (M4), clang build of ocd_native: the original values.
  windows  Windows x86-64 (i9-14900), WinLibs MinGW GCC 16 build of ocd_native (ocd_first_lap
           3945634 C++, sf70h-f1 Python). Not circular: GOLDEN_LEGACY was generated on the pre-R1
           tree 7a2bee2 and GOLDEN_PBRS on the pre-R4 tree ad3b879 (W1 proposal), and both equal
           the current tree's (e616f94) digests; same episode-end sequences as the Mac.
"""
import platform
import sys
import pytest


def platform_tag():
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        return 'mac'
    if sys.platform == 'win32' and platform.machine() in ('AMD64', 'x86_64'):
        return 'windows'
    return f'{sys.platform}-{platform.machine()}'


def golden(values):
    tag = platform_tag()
    if tag not in values:
        pytest.skip(f'no golden value recorded for platform {tag!r} (have {sorted(values)})')
    return values[tag]


# r1_golden.legacy_rollout digests (test_r1 test 1, test_r2, test_r4)
GOLDEN_LEGACY = {'mac': '0007e2813d2d28971223460790b45852513768865f08d6d0b3685f5868a1c4c2',
                 'windows': 'e92f9f9756ab664ff8b69b177e9b679f7039ce17e3ca02c75aade458609458e6'}
GOLDEN_PBRS = {'mac': '3104af11981e51be74c2a9475c402e9c273340794ccd4dc62baf836e13e9e7aa',
               'windows': 'f43009a3d025ac8041d63aa5aa88b2c448625fcaa2eb4432f50d94d91109a332'}
# f1_r4 corner table key[:12] (test_r4); it hashes the QSS key, which hashes the ocd_native binary
TABLE_KEY = {'mac': '6c61b28a4876', 'windows': '4a04e0ab3262'}
