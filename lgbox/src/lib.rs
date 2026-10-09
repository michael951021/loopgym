use std::collections::BTreeMap;
use std::io::{self, Read};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, Semaphore};

const TAIL_BYTES: usize = 4096;
const FSIZE_LIMIT: u64 = 16 * 1024 * 1024;
const NPROC_LIMIT: u64 = 64;

#[derive(Debug, Clone, Deserialize)]
pub struct RunReq {
    pub files: BTreeMap<String, String>,
    pub cmd: Vec<String>,
    pub timeout_s: u32,
    pub mem_mb: u32,
    #[serde(default)]
    pub outputs: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct RunOut {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout_tail: String,
    pub stderr_tail: String,
    pub wall_ms: u64,
    pub outputs: BTreeMap<String, Option<String>>,
}

struct Job {
    req: RunReq,
    tx: oneshot::Sender<RunOut>,
}

pub struct AppState {
    tx: mpsc::Sender<Job>,
}

/// A path is bad if it is absolute, contains `..`, or is empty.
pub fn bad_path(p: &str) -> bool {
    p.is_empty() || p.starts_with('/') || p.contains("..")
}

/// Build the app: a bounded queue of depth `queue` drained by a dispatcher
/// that caps concurrency at `workers` (semaphore). Bind the returned Router
/// to a socket of your choice (tests do this on 127.0.0.1:0).
pub fn app(workers: usize, queue: usize) -> Router {
    let (tx, rx) = mpsc::channel(queue.max(1));
    let sem = Arc::new(Semaphore::new(workers.max(1)));
    tokio::spawn(dispatch(rx, sem));
    Router::new()
        .route("/run", post(run_handler))
        .route("/health", get(|| async { "ok" }))
        .with_state(Arc::new(AppState { tx }))
}

async fn dispatch(mut rx: mpsc::Receiver<Job>, sem: Arc<Semaphore>) {
    while let Some(job) = rx.recv().await {
        let sem = sem.clone();
        tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let out = tokio::task::spawn_blocking(move || run_blocking(job.req))
                .await
                .unwrap_or_else(|e| RunOut {
                    exit_code: None,
                    timed_out: false,
                    stdout_tail: String::new(),
                    stderr_tail: format!("runner error: {e}"),
                    wall_ms: 0,
                    outputs: BTreeMap::new(),
                });
            let _ = job.tx.send(out);
        });
    }
}

fn queue_full() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, "1")],
        Json(serde_json::json!({"error": "queue full"})),
    )
        .into_response()
}

async fn run_handler(state: State<Arc<AppState>>, Json(req): Json<RunReq>) -> Response {
    for p in req.files.keys().chain(req.outputs.iter()) {
        if bad_path(p) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("bad path: {p}")})),
            )
                .into_response();
        }
    }
    if req.cmd.is_empty() || req.timeout_s == 0 || req.mem_mb == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "cmd, timeout_s and mem_mb are required"})),
        )
            .into_response();
    }
    let (tx, rx) = oneshot::channel();
    if state.tx.try_send(Job { req, tx }).is_err() {
        return queue_full();
    }
    match rx.await {
        Ok(out) => (StatusCode::OK, Json(out)).into_response(),
        Err(_) => queue_full(),
    }
}

fn run_blocking(req: RunReq) -> RunOut {
    let start = Instant::now();
    let tmp = match tempfile::tempdir() {
        Ok(t) => t,
        Err(e) => return fail(format!("tempdir: {e}"), start),
    };
    for (name, text) in &req.files {
        let p = tmp.path().join(name);
        if let Some(parent) = p.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return fail(format!("dir {name}: {e}"), start);
            }
        }
        if let Err(e) = std::fs::write(&p, text) {
            return fail(format!("write {name}: {e}"), start);
        }
    }
    let mem_bytes = req.mem_mb as u64 * 1024 * 1024;
    let cpu_s = req.timeout_s as u64 + 1;
    let mut cmd = Command::new(&req.cmd[0]);
    cmd.args(&req.cmd[1..])
        .current_dir(tmp.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // pre_exec runs in the forked child before exec: setpgid and setrlimit
    // are async-signal-safe, so this is the only safe place for them.
    unsafe {
        cmd.pre_exec(move || {
            libc::setpgid(0, 0);
            let ok = |which: libc::c_uint, v: u64| -> io::Result<()> {
                let r = libc::rlimit {
                    rlim_cur: v,
                    rlim_max: v,
                };
                if libc::setrlimit(which, &r) == 0 {
                    Ok(())
                } else {
                    Err(io::Error::last_os_error())
                }
            };
            ok(libc::RLIMIT_AS, mem_bytes)?;
            ok(libc::RLIMIT_CPU, cpu_s)?;
            ok(libc::RLIMIT_NPROC, NPROC_LIMIT)?;
            ok(libc::RLIMIT_FSIZE, FSIZE_LIMIT)?;
            Ok(())
        });
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return fail(format!("spawn: {e}"), start),
    };
    let pid = child.id() as i32;
    let out_pipe = child.stdout.take().unwrap();
    let err_pipe = child.stderr.take().unwrap();
    let out_buf = Arc::new(Mutex::new(Vec::new()));
    let err_buf = Arc::new(Mutex::new(Vec::new()));
    let (ob, eb) = (out_buf.clone(), err_buf.clone());
    let t_out = std::thread::spawn(move || read_tail(out_pipe, ob));
    let t_err = std::thread::spawn(move || read_tail(err_pipe, eb));
    let deadline = start + Duration::from_secs(req.timeout_s as u64);
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {
                if Instant::now() >= deadline {
                    unsafe {
                        libc::killpg(pid, libc::SIGKILL);
                    }
                    timed_out = true;
                    break child.wait().unwrap_or_else(|_| ExitStatus::from_raw(255));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => {
                if Instant::now() >= deadline {
                    unsafe {
                        libc::killpg(pid, libc::SIGKILL);
                    }
                    timed_out = true;
                }
                break child.wait().unwrap_or_else(|_| ExitStatus::from_raw(255));
            }
        }
    };
    t_out.join().ok();
    t_err.join().ok();
    let outputs = req
        .outputs
        .iter()
        .map(|o| {
            let v = std::fs::read(tmp.path().join(o))
                .ok()
                .map(|b| String::from_utf8_lossy(&b).into_owned());
            (o.clone(), v)
        })
        .collect();
    let out = RunOut {
        exit_code: status.code(),
        timed_out,
        stdout_tail: String::from_utf8_lossy(&out_buf.lock().unwrap()).into_owned(),
        stderr_tail: String::from_utf8_lossy(&err_buf.lock().unwrap()).into_owned(),
        wall_ms: start.elapsed().as_millis() as u64,
        outputs,
    };
    drop(tmp);
    out
}

fn fail(msg: String, start: Instant) -> RunOut {
    RunOut {
        exit_code: None,
        timed_out: false,
        stdout_tail: String::new(),
        stderr_tail: msg,
        wall_ms: start.elapsed().as_millis() as u64,
        outputs: BTreeMap::new(),
    }
}

/// Drain a pipe into `buf`, keeping only the last TAIL_BYTES.
fn read_tail(mut r: impl Read, buf: Arc<Mutex<Vec<u8>>>) {
    let mut chunk = [0u8; 8192];
    loop {
        let n = match r.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let mut b = buf.lock().unwrap();
        if n >= TAIL_BYTES {
            b.clear();
            b.extend_from_slice(&chunk[TAIL_BYTES - n..]);
        } else if b.len() + n > TAIL_BYTES {
            let dropn = b.len() + n - TAIL_BYTES;
            b.drain(..dropn);
            b.extend_from_slice(&chunk[..n]);
        } else {
            b.extend_from_slice(&chunk[..n]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_rules() {
        assert!(bad_path("../x"));
        assert!(bad_path("/abs"));
        assert!(bad_path("a/../b"));
        assert!(bad_path(""));
        assert!(!bad_path("ok.txt"));
        assert!(!bad_path("dir/nested.txt"));
        assert!(bad_path("a..b")); // spec: paths containing `..` are rejected
    }
}
