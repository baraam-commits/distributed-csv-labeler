import argparse, csv, hashlib, json, os, socket, sys, threading, time, glob, datetime
from typing import List, Optional, Dict
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import signal

# ---------------- Tunables via env ----------------
DEFAULT_BATCH = int(os.getenv("BATCH", "128"))
HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_SEC", "1.0"))
STALE_SEC     = float(os.getenv("STALE_SEC", "5.0"))
REPL_INTERVAL = float(os.getenv("REPL_INTERVAL", "10.0"))
PEER_TIMEOUT  = float(os.getenv("PEER_TIMEOUT", "2.0"))

# ---------------- CLI ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=os.getenv("CSV", "/data/questions.csv"))
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8001")))
parser.add_argument("--peers", default=os.getenv("PEERS", "node1:8001"))
parser.add_argument("--worker-id", default=os.getenv("WORKER_ID"))
parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
parser.add_argument("--mode", choices=["auto","server","client"], default=os.getenv("MODE","auto"))
parser.add_argument("--prefer-leader", action="store_true")
args = parser.parse_args()

WORKER_ID = args.worker_id or f"{socket.gethostname()}:{args.port}"
PEERS: List[str] = [p.strip() for p in args.peers.split(",") if p.strip()]
CSV_PATH = args.csv
BATCH_SIZE = args.batch

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/output")
STATE_DIR  = os.getenv("STATE_DIR",  "/state")
REPL_DIR   = os.getenv("REPL_DIR",   "/replicated")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR,  exist_ok=True)
os.makedirs(REPL_DIR,   exist_ok=True)

RESULTS_BASENAME = f"labels_{WORKER_ID.replace(':','_')}"
STATE_PATH       = os.path.join(STATE_DIR,  f"state_{WORKER_ID.replace(':','_')}.json")

# -------------- Graceful stop ---------------------
stop_flag = {"stop": False}
def handle_sig(*_): stop_flag["stop"] = True
signal.signal(signal.SIGINT, handle_sig)
signal.signal(signal.SIGTERM, handle_sig)

# -------------- Load CSV --------------------------
DATA: List[Dict[str,str]] = []
try:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "text" not in reader.fieldnames:
            print("CSV must have a 'text' column.", file=sys.stderr); sys.exit(1)
        for row in reader:
            txt = (row["text"] or "").strip()
            _id = hashlib.sha1(txt.encode("utf-8")).hexdigest()
            DATA.append({"id": _id, "text": txt})
except FileNotFoundError:
    print(f"CSV not found at {CSV_PATH}", file=sys.stderr); sys.exit(1)
N = len(DATA)
print(f"[{WORKER_ID}] loaded {N} rows")

# -------------- State -----------------------------
state_lock = threading.Lock()
state = {
    "worker_id": WORKER_ID,
    "current_index": 0,    # furthest end+1 processed locally
    "epoch": 0,
    "leader": False,
    "last_heartbeat": 0.0,
    "known_leader": None,
    "next_index": 0,       # leader-only pointer
}

def save_state():
    with state_lock:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                s = json.load(f)
            with state_lock:
                state.update(s)
            print(f"[{WORKER_ID}] restored state: {state}")
        except Exception as e:
            print(f"[{WORKER_ID}] failed to load state: {e}")
load_state()

# -------------- Shards (minute rotation) ----------
def minute_stamp(ts=None):
    dt = datetime.datetime.utcfromtimestamp(ts or time.time())
    return dt.strftime("%Y%m%dT%H%M")  # minute granularity

def shard_name(worker_id: str, stamp: str):
    wid = worker_id.replace(":","_")
    return f"labels_{wid}_{stamp}.jsonl"

def current_local_shard_path() -> str:
    return os.path.join(OUTPUT_DIR, shard_name(WORKER_ID, minute_stamp()))

def list_completed_local_shards() -> List[str]:
    cur = os.path.basename(current_local_shard_path())
    files = [os.path.basename(p) for p in glob.glob(os.path.join(OUTPUT_DIR, "labels_*.jsonl"))]
    return sorted([f for f in files if f != cur])

# -------------- API schema ------------------------
class Status(BaseModel):
    worker_id: str
    current_index: int
    epoch: int
    leader: bool
    ts: float

class ClaimResp(BaseModel):
    epoch: int
    start: int
    end: int

# -------------- FastAPI app -----------------------
app = FastAPI()

@app.get("/ping")
def ping():
    with state_lock:
        return {"ok": True, "epoch": state["epoch"], "leader": state["leader"], "id": state["worker_id"]}

@app.get("/status", response_model=Status)
def get_status():
    with state_lock:
        return Status(worker_id=state["worker_id"], current_index=state["current_index"],
                      epoch=state["epoch"], leader=state["leader"], ts=time.time())

@app.post("/claim", response_model=ClaimResp)
def claim():
    with state_lock:
        if not state["leader"]:
            raise HTTPException(status_code=423, detail="Not leader")
        start = state["next_index"]
        if start >= N:
            raise HTTPException(status_code=204, detail="No work")
        end = min(N - 1, start + BATCH_SIZE - 1)
        state["next_index"] = end + 1
        save_state()
        print(f"[{WORKER_ID}] claim -> {start}-{end}")
        return ClaimResp(epoch=state["epoch"], start=start, end=end)

# advertise completed shards only (immutable)
@app.get("/shards")
def shards():
    return {"files": list_completed_local_shards()}

# allow streaming shard by name
@app.get("/pull")
def pull(name: str):
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="bad name")
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")
    def it():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024*256)
                if not chunk: break
                yield chunk
    return StreamingResponse(it(), media_type="application/octet-stream")

# -------------- Gossip & election -----------------
peer_status: Dict[str, Status] = {}

def poll_peer_status(peer_url: str):
    try:
        with httpx.Client(timeout=PEER_TIMEOUT) as cli:
            r = cli.get(f"http://{peer_url}/status")
            if r.status_code == 200:
                st = Status(**r.json())
                peer_status[st.worker_id] = st
    except Exception:
        pass

def alive_peers() -> Dict[str, Status]:
    now = time.time()
    return {wid: st for wid, st in peer_status.items() if now - st.ts <= STALE_SEC}

def elect_leader() -> str:
    alive = alive_peers()
    candidates = list(alive.values())
    with state_lock:
        me = Status(worker_id=state["worker_id"], current_index=state["current_index"],
                    epoch=state["epoch"], leader=state["leader"], ts=time.time())
    candidates.append(me)
    winner = min(candidates, key=lambda s: (s.current_index, s.worker_id))
    return winner.worker_id

def heartbeat_loop():
    if args.mode == "auto" and args.prefer_leader:
        with state_lock:
            state["leader"] = True
            state["epoch"] += 1
            state["next_index"] = state["current_index"]
            print(f"[{WORKER_ID}] prefer-leader boot; epoch={state['epoch']}")
            save_state()
    while not stop_flag["stop"]:
        if args.mode in ("auto","client"):
            for p in PEERS: poll_peer_status(p)
        now = time.time()
        if args.mode == "auto":
            winner = elect_leader()
            with state_lock:
                was_leader = state["leader"]
                if winner == state["worker_id"]:
                    if not state["leader"]:
                        state["leader"] = True
                        state["epoch"] += 1
                        ap = alive_peers().values()
                        min_idx = min([state["current_index"]] + [st.current_index for st in ap]) if ap else state["current_index"]
                        state["next_index"] = min_idx
                        print(f"[{WORKER_ID}] PROMOTED leader; epoch={state['epoch']} next_index={state['next_index']}")
                        save_state()
                else:
                    if was_leader: print(f"[{WORKER_ID}] DEMOTED (winner {winner})")
                    state["leader"] = False
                    state["known_leader"] = winner
                state["last_heartbeat"] = now
        elif args.mode == "server":
            with state_lock:
                if not state["leader"]:
                    state["leader"] = True
                    state["epoch"] += 1
                    state["next_index"] = state["current_index"]
                    print(f"[{WORKER_ID}] MANUAL leader; epoch={state['epoch']}")
                    save_state()
                state["last_heartbeat"] = now
        else:  # client
            winner = elect_leader()
            with state_lock:
                state["leader"] = False
                state["known_leader"] = winner
                state["last_heartbeat"] = now
        time.sleep(HEARTBEAT_SEC)

# -------------- Worker + labeling -----------------
def process_range(start: int, end: int):
    shard_path = current_local_shard_path()
    with open(shard_path, "a", encoding="utf-8") as out:
        for idx in range(start, end + 1):
            rec = DATA[idx]
            txt = rec["text"]
            # TODO: replace with your real LLM/rules
            label = "search" if ("?" in txt or any(k in txt.lower() for k in [" who"," when"," where"," what"," define "])) else "no-search"
            conf = 0.9 if label == "search" else 0.8
            
            out.write(json.dumps({
                "id": rec["id"], "idx": idx,
                "label": label, "confidence": conf,
                "worker": WORKER_ID, "ts": time.time()
            }, ensure_ascii=False) + "\n")
            out.flush()
    with state_lock:
        state["current_index"] = max(state["current_index"], end + 1)
        save_state()

def worker_loop():
    while not stop_flag["stop"]:
        wid_to_url: Dict[str,str] = {}
        for p in PEERS:
            try:
                with httpx.Client(timeout=PEER_TIMEOUT) as cli:
                    r = cli.get(f"http://{p}/status")
                    if r.status_code == 200:
                        st = Status(**r.json()); wid_to_url[st.worker_id] = p; peer_status[st.worker_id] = st
            except Exception: pass

        with state_lock:
            i_am_leader = state["leader"]
        if args.mode == "client": i_am_leader = False

        if i_am_leader:
            url = f"http://127.0.0.1:{args.port}"
        else:
            winner = elect_leader() if args.mode != "server" else state.get("known_leader")
            if not winner or winner not in wid_to_url:
                time.sleep(0.5); continue
            url = f"http://{wid_to_url[winner]}"

        try:
            with httpx.Client(timeout=5.0) as cli:
                r = cli.post(f"{url}/claim")
                if r.status_code in (204, 423):
                    time.sleep(0.5); continue
                r.raise_for_status()
                payload = ClaimResp(**r.json()); start, end = payload.start, payload.end
        except Exception:
            time.sleep(0.5); continue

        process_range(start, end)
        time.sleep(0.01)

# -------------- Replication -----------------------
def list_peer_shards(peer_url: str) -> List[str]:
    try:
        with httpx.Client(timeout=PEER_TIMEOUT) as cli:
            r = cli.get(f"http://{peer_url}/shards")
            if r.status_code == 200:
                return r.json().get("files", [])
    except Exception: pass
    return []

def download_peer_shard(peer_url: str, name: str):
    if os.path.exists(os.path.join(OUTPUT_DIR, name)) or os.path.exists(os.path.join(REPL_DIR, name)):
        return
    try:
        with httpx.Client(timeout=None) as cli:
            with cli.stream("GET", f"http://{peer_url}/pull", params={"name": name}) as r:
                r.raise_for_status()
                tmp = os.path.join(REPL_DIR, name + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1024*256):
                        if chunk: f.write(chunk)
                os.replace(tmp, os.path.join(REPL_DIR, name))
                print(f"[{WORKER_ID}] replicated {name} from {peer_url}")
    except Exception: pass

def replication_loop():
    while not stop_flag["stop"]:
        peer_map = {p: list_peer_shards(p) for p in PEERS}
        names = set()
        for files in peer_map.values(): names.update(files)
        for name in sorted(names):
            if name.startswith(f"labels_{WORKER_ID.replace(':','_')}"): continue
            for p, files in peer_map.items():
                if name in files:
                    download_peer_shard(p, name); break
        time.sleep(REPL_INTERVAL)

# -------------- Startup ---------------------------
def main():
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=replication_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
if __name__ == "__main__": main()
