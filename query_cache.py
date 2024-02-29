import uuid
import os
from typing import Optional, List
import time
from helper_utils import targz, untargz

CACHE_LOC = "api_query_cache"


class UniqId:
    def __init__(self, txt: List[str]):
        s = "\n".join(txt)
        # uuid3 does a MD5 hash; use uuid5 if you need SHA1
        self.uuid = uuid.uuid3(uuid.NAMESPACE_OID, s)

    def __repr__(self) -> str:
        return str(self.uuid)


class QueryCache:
    def __init__(self, name):
        self.dir = os.path.join(CACHE_LOC, name)
        self.ensure_cache_dir()

    def ensure_cache_dir(self):
        self.ensure_exists(CACHE_LOC)
        self.load_snapshot_or_empty()

    def load_snapshot_or_empty(self):
        snapshot = self.snapshot_filename()
        if not os.path.exists(self.dir):
            if os.path.exists(snapshot) and self.confirm(f'Cache snapshot {snapshot} exists. Initialize from it (highly recommended)'):
                self.load_snapshot()
            else:
                self.ensure_exists(self.dir)
        return

    def save_snapshot(self):
        save_loc = self.snapshot_filename()
        do_save = self.confirm(f"Save query snapshot to {save_loc}")
        if not do_save:
            return
        targz(self.dir)
        assert os.path.exists(save_loc), f'Save should have created {save_loc}. Did not.'

    def load_snapshot(self):
        untargz(self.snapshot_filename(), self.dir)
        assert os.path.exists(self.dir), f'Load should have created {self.dir}. Did not.'

    def confirm(self, msg: str) -> bool:
        do = input(msg + "[y]/n? ")
        return do != "n"

    def snapshot_filename(self):
        return str(self.dir) + ".tar.gz"

    def ensure_exists(self, d):
        if not os.path.exists(d):
            os.mkdir(d)

    def put(self, q: str, a: str) -> bool:
        old = self.get(q)
        if old is not None and old != a:
            print(f'[WARN] Cache already has map for "{q[:20]}.." -> "{old[:20]}..", but asking to overwrite with different {a[:20]}.. Overwrite ignored!')
            return False
        uuid = UniqId([q])
        self.cache_write(uuid, a)
        return True

    def get(self, q: str) -> Optional[str]:
        uuid = UniqId([q])
        return self.cache_get(uuid) if self.cache_exists(uuid) else None

    def cache_get(self, uuid: UniqId) -> str:
        with open(self.fname(uuid), 'r') as f:
            return f.read()

    def cache_write(self, uuid: UniqId, val: str) -> None:
        with open(self.fname(uuid), 'w') as f:
            f.write(val)

    def cache_exists(self, uuid: UniqId) -> bool:
        return os.path.exists(self.fname(uuid))

    def fname(self, uuid: UniqId) -> str:
        return os.path.join(self.dir, f'{uuid}')

    def purge(self):
        for c in os.listdir(self.dir):
            cache_file = os.path.join(self.dir, c)
            print(f"Purging {cache_file}")
            time.sleep(1)
            os.remove(cache_file)


def test_cache_write() -> None:
    q12 = "question 1 2"
    q23 = "question 2 3"
    a12 = "answer 1 2"
    a23 = "answer 2 3"
    cache = QueryCache("TEST")

    cache.purge()

    # empty cache
    assert cache.get(q12) == None
    assert cache.get(q23) == None

    # put a value and retrieve it
    assert cache.put(q12, a12) == True
    assert cache.get(q12) == a12

    # put new cache value, make sure old value did not change
    assert cache.put(q23, a23) == True
    assert cache.get(q23) == a23
    assert cache.get(q12) == a12

    # fail to overwrite
    assert cache.put(q23, a12) == False
    assert cache.get(q23) == a23
