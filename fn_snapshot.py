from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from helper_utils import download_extract
from typing import Dict, Any, List, Tuple, Optional
import json

DATE_FMT = '%b-%Y'
STATIC_DATE_TAG = "Jan-1984"

# MATH/test has the subject dirs within it
root_test = lambda root: os.path.join(root, "test")

class FnSnapshot:
    def __init__(self, date: str, url: str):
        self.date = date
        self.url = url

    def is_static(self) -> bool:
        return self.date == STATIC_DATE_TAG

    def ensure_dataset(self):
        # get root "MATH" from "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"
        # or "Dec-2023" from "Dec-2023.tar.gz"
        p = Path(self.url)
        root = p.name.removesuffix("".join(p.suffixes))

        downloaded_name = os.path.basename(self.url)
        is_gz = downloaded_name.endswith('gz')
        there = lambda d: os.path.exists(d)
        test_dir = root_test(root)
        if not there(root) or not there(test_dir):
            download_extract(self.url, root, downloaded_name, root, is_gz)
            assert there(root), f'Download/extract failed to create {root} from {self.url}'
            assert there(test_dir), f'Download/extract from {self.url} does not have {test_dir}'
        return root

    @classmethod
    def load(cls, benchmark: str, config: str) -> List[FnSnapshot]:
        def load_snaps(fixed_url, fns):
            snaps = [FnSnapshot(fn['date'], fn['url']) for fn in fns]
            snaps.append(FnSnapshot(STATIC_DATE_TAG, fixed_url))
            snaps.sort(key = lambda x: datetime.strptime(x.date, DATE_FMT))
            return snaps

        with open(config, 'r') as cf:
            benchmarks = json.load(cf)
        benchmark_meta = None
        for meta in benchmarks:
            if meta['benchmark'] == benchmark:
                benchmark_meta = meta
        assert benchmark_meta, f'Did not find {benchmark} in {benchmarks}'
        return load_snaps(benchmark_meta['fixed'], benchmark_meta['functionals'])

