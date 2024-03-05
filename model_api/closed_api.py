from typing import Dict, List, Tuple, Callable, Any, Awaitable, Optional
import time
import sys
import os
import asyncio
import math
import functools
from codetiming import Timer # type: ignore
from tqdm import tqdm # type: ignore
from query_cache import QueryCache
from persist import Persist
from enum import Enum

# temperature threshold under which we expect deterministic answers
# the queries will be cached and we won't make API calls to OAI
# If temp is above the threshold then every time we will API to OAI
DETERMINISTIC_THRESHOLD = 1.0

NOW_MS = lambda: round(time.time() * 1000)
PERIODS_PER_MIN = 4
ONE_PERIOD_MS = 1000 * 60 / PERIODS_PER_MIN

# If ServiceUnavailableError, or Timeout etc
BACKOFF_NUM_RETRIES = 3
BACKOFF_TIME_SECONDS = 120
hr = "=" * 100
BACKOFF_MSG = lambda e: f'\n\n{hr}\nService unavailable.\nError = {e}\nPausing for {BACKOFF_TIME_SECONDS} seconds.\n{hr}\n\n'
BACKOFF_ABORT_MSG = lambda e: f'\n\n{hr}\nService unavailable after {BACKOFF_NUM_RETRIES} attempts.\nError = {e}\n{hr}\nAborting!\n\n'

class KnownModel:
    def __init__(self,
            name: str,
            is_chat: bool,
            api_org: Optional[str],
            api_key: Optional[str],
            api_base: str,
            stops: Optional[List[str]] = None,
            prompt_format: Optional[str] = None,
            server_path: Optional[str] = None,
            batch_size: int = 200,
            num_parallel: int = 50,
            cps: float = 0.0,
            ):
        self.name = name
        self.is_chat = is_chat
        self.api_org = api_org
        self.api_key = api_key
        self.api_base = api_base
        self.server_path = server_path

        self.stops = stops
        self.prompt_format = prompt_format

        self.batch_size = batch_size
        self.num_parallel = num_parallel
        self.cost_per_token = cps

    def __hash__(self):
        hashes = [hash(v) for v in vars(self)]
        init = 42
        return functools.reduce(lambda a, b: a ^ b, hashes, init)

    def __eq__(self, other):
        eq_vars = vars(self) == vars(other)
        return (isinstance(other, self.__class__) and eq_vars)

class QStat:
    def __init__(self, cost, time_ms):
        self.cost = cost
        self.time_ms = time_ms
        self.timestamp = NOW_MS()

class QueryStats:
    def __init__(self):
        self.stats: List[QStat] = []

    def log_queries(self, qs: List[QStat]):
        self.stats += qs

    def __repr__(self) -> str:
        total = len(self.stats)
        per_query_spend = 0.0
        per_query_ms = 0.0
        queries_min = 0.0

        if total > 0:
            spend = sum(q.cost for q in self.stats)
            ms = sum(q.time_ms for q in self.stats)
            per_query_spend = float(spend) / total
            per_query_ms = float(ms) / total

            # estimate q/min
            max_ts = max(q.timestamp for q in self.stats)
            min_ts = min(q.timestamp for q in self.stats)
            minutes = (max_ts - min_ts) / (60 * 1000.0)
            ms_per_query = ( max_ts - min_ts ) / total
            queries_min = PERIODS_PER_MIN * ONE_PERIOD_MS / ms_per_query if ms_per_query else 0.0

        return f'{total} queries; {minutes:.2f} mins total time; per query = ${per_query_spend:.4f} {per_query_ms / 1000.0:.2f}s; rate = {queries_min:.0f}queries/min, 0 time/cost implies cache lookup]'

ResponseStats = Tuple[str, QStat]
WorkFn = Callable[[Any], Awaitable[Tuple[ResponseStats, bool]]]

class Workers:
    def __init__(self, async_fn: WorkFn, num_parallel):
        self.async_fn = async_fn
        self.num_parallel = num_parallel

    def do(self, tasks) -> Tuple[List[ResponseStats], int]:
        return asyncio.run(self.parallel(tasks))

    async def work(self, worker_ident, in_q, out_q):
        while not in_q.empty():
            tid, task = await in_q.get()
            rslt, is_expensive = await self.async_fn(task)
            await out_q.put((tid, is_expensive, rslt))

    async def progress(self, out_q, sz):
        pbar = tqdm(total = sz)
        out_sz = 0
        while out_sz < sz:
            new_sz = out_q.qsize()
            update = new_sz - out_sz
            out_sz = new_sz
            pbar.update(update)
            await asyncio.sleep(0.5)
        pbar.close()

    async def parallel(self, tasks) -> Tuple[List[ResponseStats], int]:
        # Create the queue of work
        in_q: asyncio.Queue[Tuple[int, Any]] = asyncio.Queue()
        out_q: asyncio.Queue[Tuple[int, bool, Any]] = asyncio.Queue()

        # Put some work in the queue
        for tid, t in enumerate(tasks):
            await in_q.put((tid, t))

        # create the deferred workers
        create = lambda x: asyncio.create_task(self.work(x, in_q, out_q))
        workers = [create(i) for i in range(self.num_parallel)]
        workers.append(asyncio.create_task(self.progress(out_q, len(tasks))))

        # Run the tasks
        await asyncio.gather(*workers)

        results = []
        while not out_q.empty():
            results.append(await out_q.get())
        results.sort(key = lambda x: x[0])
        ids, expensive_flags, done = list(zip(*results))
        assert list(ids) == sorted(list(ids))
        expensed = expensive_flags.count(True)
        return list(done), expensed

class LogLevel(Enum):
    ERROR = 3
    WARN = 2
    INFO = 1

    LOWEST = 1

class RateLimited:
    def __init__(self, async_fn: WorkFn, batch_size: int, num_parallel: int, console_log_lvl: LogLevel):
        self.workers = Workers(async_fn, num_parallel)
        self.overall_stats = QueryStats()
        self.batch_size = batch_size
        self.console_log_lvl = console_log_lvl

    def do(self, tasks) -> List[str]:
        with Timer(text="Total elapsed time: {:.1f}"):
            chunks = self.chunks(tasks, self.batch_size)
            num_chunks = math.ceil(len(tasks) / self.batch_size)
            responses: List[ResponseStats] = self.do_one_per_period(chunks, num_chunks, self.workers.do)
            answers, stats = list(zip(*responses))
            self.overall_stats.log_queries(list(stats))
        self.console_log(LogLevel.INFO, f'stats: {self.overall_stats}')
        return list(answers)

    def do_one_per_period(self, per_period_chunks, num_chunks,
            per_period_fn: Callable[[Any], Tuple[List[ResponseStats], int]]) -> List[ResponseStats]:
        responses: List[ResponseStats] = []
        for count, chunk in enumerate(per_period_chunks):
            with Timer(text="Expected to spend 60s if API called. Elapsed: {:.1f}s. (< 60s ok, if cache lookup)"):
                self.console_log(LogLevel.INFO, f'Sending {len(chunk)} queries to OAI in parallel.')
                start = NOW_MS()
                resp_list, num_expensive = self.with_unavailable_backoff(per_period_fn, chunk)
                responses += resp_list
                elapsed = NOW_MS() - start
                frac_api = float(num_expensive) / len(chunk)
                frac_cache = float(len(chunk) - num_expensive) / len(chunk)
                self.console_log(LogLevel.INFO, f'Done {count + 1}/{num_chunks} ({frac_api * 100:.2f}% API calls, {frac_cache * 100:.2f}% cache lookup)')

                # sleep for 100ms more than what is needed to make it one min
                pause_ms = frac_api * (ONE_PERIOD_MS - elapsed + 100)
                if pause_ms > 0:
                    self.console_log(LogLevel.WARN, f'Pausing for {pause_ms / 1000.0}s to stay within API rate budget')
                    time.sleep(pause_ms / 1000.0)
        return responses

    def console_log(self, lvl: LogLevel, msg):
        if lvl.value >= self.console_log_lvl.value:
            return
        print(msg)

    def with_unavailable_backoff(self, per_period_fn, chunk):
        tries = 0
        while tries < BACKOFF_NUM_RETRIES:
            try:
                return per_period_fn(chunk)
            except Exception as e:
                self.console_log(LogLevel.WARN, BACKOFF_MSG(e))
                time.sleep(BACKOFF_TIME_SECONDS)
                ex = e
            tries += 1
        # if failed after enough tries, terminate process
        # restarting will pull any data already downloaded from cache
        # so no destructive loss. we just revert to user to manually restart
        self.console_log(LogLevel.WARN, BACKOFF_ABORT_MSG(ex))
        raise ex

    def chunks(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]


class ParameterizedModel:
    def __init__(self,
            temp: float,
            instruction: str,
            who_are_you: str,
            few_shot: Dict[str, str],
            model: KnownModel,
            agent_name: str,
            mk_tasks: bool = False,
            batch_sz: int = 200,
            num_parallel: int = 50,
            ):
        self.pull_from_cache = temp < DETERMINISTIC_THRESHOLD

        self.temperature = temp
        self.instruction = instruction
        self.who_are_you = who_are_you
        self.few_shot = few_shot
        self.model = model

        self.agent_name = agent_name
        self.batch_sz = batch_sz
        self.num_parallel = num_parallel

        self.mk_tasks = mk_tasks

    def __hash__(self):
        hashes = [hash(v) for v in vars(self)]
        init = 42
        return functools.reduce(lambda a, b: a ^ b, hashes, init)

    def __eq__(self, other):
        eq_vars = vars(self) == vars(other)
        return (isinstance(other, self.__class__) and eq_vars)

    def __repr__(self):
        return f'PM({self.model.name}, {self.agent_name}, task_maker = {self.mk_tasks})'

class TasksForModel:
    def __init__(self, tasks: List[str], params: ParameterizedModel):
        self.tasks = tasks
        self.params = params

class OracleTaskQ:
    TASKS_DIR = "oracle_task_queue"
    TASK_FILE_SUFFIX = ".pickle"

    @classmethod
    def write_tasks(cls, tasks: List[str], params: ParameterizedModel) -> None:
        cls.ensure_tq_dir_exists()
        existing: List[int] = cls.idents_in_task_dir()
        ident = max(existing) + 1 if existing else 0
        fname = cls.task_file(ident)
        tm = TasksForModel(tasks, params)
        Persist.save(tm, fname)

    @classmethod
    def task_file(cls, ident: int) -> str:
        cls.ensure_tq_dir_exists()
        return os.path.join(cls.TASKS_DIR, f"{ident}{cls.TASK_FILE_SUFFIX}")

    @classmethod
    def idents_in_task_dir(cls) -> List[int]:
        cls.ensure_tq_dir_exists()
        extract = lambda fname: int(fname[:-len(cls.TASK_FILE_SUFFIX)])
        return sorted([extract(f) for f in os.listdir(cls.TASKS_DIR)])

    @classmethod
    def purge_tasks(cls, idents: List[int]) -> None:
        cls.ensure_tq_dir_exists()
        for i in idents:
            fname = cls.task_file(i)
            os.remove(fname)

    @classmethod
    def ensure_tq_dir_exists(cls) -> None:
        if not os.path.exists(cls.TASKS_DIR):
            os.makedirs(cls.TASKS_DIR, exist_ok = True)

class ClosedAPI:
    MAKER_PLACEHOLDER = "PLACEHOLDER_FILL_LATER"

    def __init__(self, params: ParameterizedModel, console_loglevel: LogLevel = LogLevel.LOWEST):
        self.params = params
        self.cacher = QueryCache(params.agent_name)
        self.rate_limited_runner = RateLimited(self.cached_api_call, params.batch_sz, params.num_parallel, console_loglevel)
        self.TASK_MAKER_MODE = params.mk_tasks

    # to be overriden
    def deobject(self, obj_resp: Any) -> str: # type: ignore [empty-body]
        pass

    # to be overriden
    def extract_answer(self, resp: str) -> str: # type: ignore [empty-body]
        pass

    # to be overriden
    def to_prompt_task(self, task: str) -> Any: # type: ignore [empty-body]
        pass

    # to be overriden
    def reset_library(self, model: KnownModel) -> None: # type: ignore [empty-body]
        pass

    # to be overriden
    def model_ask(self, prompt: str) -> Awaitable[Any]: # type: ignore [empty-body]
        pass

    # to be overriden
    def spent(self, response) -> float: # type: ignore [empty-body]
        pass

    def sync_solver(self, task: str):
        ans_stat, api_call_not_cache = asyncio.run(self.cached_api_call(task))
        ans, stat = ans_stat
        return ans

    def maker_placeholder(self, sz: int) -> List[str]:
        return [ ClosedAPI.MAKER_PLACEHOLDER ] * sz

    def make_tasks(self, tasks: List[str]) -> List[str]:
        OracleTaskQ.write_tasks(tasks, self.params)
        placeholders = self.maker_placeholder(len(tasks))
        return placeholders

    def query(self, tasks: List[str], rate_limited: bool = True) -> List[str]:
        if self.TASK_MAKER_MODE:
            return self.make_tasks(tasks)

        # actual solving mode, call API
        self.reset_library(self.params.model)

        if rate_limited:
            solved = self.rate_limited_runner.do(tasks)
        else:
            solved = [self.sync_solver(s) for s in tasks]
        return solved

    async def cached_api_call(self, task: str) -> Tuple[ResponseStats, bool]:
        # if in cache pull from cache
        if self.params.pull_from_cache:
            cached = self.cacher.get(task)
            if cached:
                return (cached, QStat(0, 0)), False

        # not cached, make api call
        answer, stat = await self.api_call(task)

        # write to cache even in the case when we don't pull from it
        self.cacher.put(task, answer)

        return (answer, stat), True

    async def api_call(self, task: str) -> ResponseStats:

        start_ms = NOW_MS()
        response = await self.model_ask(self.to_prompt_task(task))
        elapsed = NOW_MS() - start_ms

        cost = self.spent(response)

        txt_response = self.deobject(response)
        return txt_response, QStat(cost, elapsed)

    def snapshot_api_query_cache(self):
        self.cacher.save_snapshot()

class Runner:
    @classmethod
    def run(cls, name: str, snapshots_specs: str,
            cot: bool, few_shot_num: int, temperature: float,
            verbose: bool = False, save_snaphot: bool = False): # type: ignore [empty-body]
        pass
