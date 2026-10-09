use std::collections::HashMap;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn start(workers: usize, queue: usize) -> u16 {
    let app = lgbox::app(workers, queue);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    for _ in 0..100 {
        if http(port, "GET", "/health", None).await.0 == 200 {
            return port;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("server did not come up");
}

/// Minimal HTTP/1.1 client over a fresh connection (connection: close).
async fn http(
    port: u16,
    method: &str,
    path: &str,
    body: Option<String>,
) -> (u16, HashMap<String, String>, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let body = body.unwrap_or_default();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n").expect("no response head");
    let mut lines = head.split("\r\n");
    let status: u16 = lines
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let mut headers = HashMap::new();
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    (status, headers, rest.to_string())
}

fn run_body(
    files: Vec<(String, String)>,
    cmd: Vec<String>,
    timeout_s: u32,
    mem_mb: u32,
    outputs: Vec<String>,
) -> String {
    let f: Vec<String> = files
        .into_iter()
        .map(|(k, v)| format!("{:?}:{}", k, serde_json::to_string(&v).unwrap()))
        .collect();
    let c: Vec<String> = cmd.into_iter().map(|x| format!("{x:?}")).collect();
    let o: Vec<String> = outputs.into_iter().map(|x| format!("{x:?}")).collect();
    format!(
        "{{\"files\":{{{}}},\"cmd\":[{}],\"timeout_s\":{timeout_s},\"mem_mb\":{mem_mb},\"outputs\":[{}]}}",
        f.join(","),
        c.join(","),
        o.join(",")
    )
}

/// Pids whose /proc/<pid>/cmdline contains `needle` (NULs kept in the string).
fn pids_with(needle: &str) -> Vec<u32> {
    let mut v = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return v;
    };
    for e in entries.flatten() {
        let Ok(name) = e.file_name().into_string() else {
            continue;
        };
        if !name.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let Ok(cmd) = std::fs::read(format!("/proc/{name}/cmdline")) else {
            continue;
        };
        if std::str::from_utf8(&cmd)
            .map(|c| c.contains(needle))
            .unwrap_or(false)
        {
            v.push(name.parse().unwrap());
        }
    }
    v
}

fn alive(p: u32) -> bool {
    unsafe { libc::kill(p as i32, 0) == 0 }
}

/// True once no pid matching `needle` with kill -0 == success remains.
async fn wait_gone(needle: &str) -> bool {
    for _ in 0..60 {
        if !pids_with(needle).iter().any(|p| alive(*p)) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

/// True once no NEW pid (vs. `before`) matching `needle` survives.
async fn new_pids_gone(needle: &str, before: &[u32]) -> bool {
    for _ in 0..60 {
        let now = pids_with(needle);
        if now.iter().all(|p| before.contains(p)) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

#[tokio::test]
async fn health_ok() {
    let port = start(1, 1).await;
    assert_eq!(http(port, "GET", "/health", None).await.0, 200);
}

#[tokio::test]
async fn echo_roundtrip_and_outputs() {
    let port = start(2, 2).await;
    let body = run_body(
        vec![(
            "a.txt".into(),
            "hello\n".into(),
        )],
        vec!["/bin/echo".into(), "hi".into()],
        10,
        256,
        vec!["a.txt".into(), "missing.txt".into()],
    );
    let (st, _, resp) = http(port, "POST", "/run", Some(body)).await;
    assert_eq!(st, 200, "resp={resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["exit_code"], 0);
    assert_eq!(v["timed_out"], false);
    assert_eq!(v["stdout_tail"], "hi\n");
    assert_eq!(v["outputs"]["a.txt"], "hello\n");
    assert!(v["outputs"]["missing.txt"].is_null());
    assert!(v["wall_ms"].as_u64().unwrap() < 2000);
}

#[tokio::test]
async fn timeout_kills_group_no_survivors() {
    let port = start(1, 4).await;
    let before: Vec<u32> = pids_with("sleep\030\0");
    let body = run_body(
        vec![],
        vec!["/bin/sleep".into(), "30".into()],
        1,
        256,
        vec![],
    );
    let t0 = Instant::now();
    let (st, _, resp) = http(port, "POST", "/run", Some(body)).await;
    let wall = t0.elapsed();
    assert_eq!(st, 200, "resp={resp}");
    assert!(wall >= Duration::from_millis(900), "early: {wall:?} resp={resp}");
    assert!(wall < Duration::from_secs(3), "slow: {wall:?}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["timed_out"], true);
    assert!(v["exit_code"].is_null());
    assert!(
        new_pids_gone("sleep\030\0", &before).await,
        "sleep child survived"
    );
    assert_eq!(http(port, "GET", "/health", None).await.0, 200);
}

#[tokio::test]
async fn fork_bomb_contained() {
    let port = start(1, 2).await;
    let tag = format!("lb-bomb-{}", std::process::id());
    let script = "import os, sys\nwhile True:\n    try:\n        pid = os.fork()\n    except OSError:\n        sys.exit(0)\n    if pid == 0:\n        sys.exit(0)\n";
    let body = run_body(
        vec![(tag.clone(), script.to_string())],
        vec!["python3".into(), tag.clone()],
        20,
        512,
        vec![],
    );
    let t0 = Instant::now();
    let (st, _, resp) = http(port, "POST", "/run", Some(body)).await;
    assert_eq!(st, 200, "resp={resp}");
    let wall = t0.elapsed();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["timed_out"], false, "bomb must end by its own EAGAIN, not timeout");
    assert_eq!(v["exit_code"], 0);
    assert!(wall < Duration::from_secs(20));
    assert!(wait_gone(&tag).await, "stray bomb processes remain");
    assert_eq!(http(port, "GET", "/health", None).await.0, 200);
}

#[tokio::test]
async fn big_alloc_fails_under_as_limit() {
    let port = start(1, 2).await;
    let tag = format!("lb-big-{}", std::process::id());
    let script =
        "import sys\ntry:\n    b = bytearray(2 * 1024 * 1024 * 1024)\nexcept MemoryError:\n    sys.exit(3)\nsys.exit(1)\n";
    let body = run_body(
        vec![(tag.clone(), script.to_string())],
        vec!["python3".into(), tag.clone()],
        20,
        256,
        vec![],
    );
    let (st, _, resp) = http(port, "POST", "/run", Some(body)).await;
    assert_eq!(st, 200, "resp={resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["timed_out"], false);
    assert_eq!(v["exit_code"], 3, "2 GB alloc must hit RLIMIT_AS: {resp}");
    assert!(wait_gone(&tag).await);
    assert_eq!(http(port, "GET", "/health", None).await.0, 200);
}

#[tokio::test]
async fn bad_paths_rejected_400() {
    let port = start(1, 1).await;
    let cases = vec![
        "{\"files\":{\"../x\":\"1\"},\"cmd\":[\"/bin/echo\"],\"timeout_s\":5,\"mem_mb\":64,\"outputs\":[]}",
        "{\"files\":{\"/abs.txt\":\"1\"},\"cmd\":[\"/bin/echo\"],\"timeout_s\":5,\"mem_mb\":64,\"outputs\":[]}",
        "{\"files\":{},\"cmd\":[\"/bin/echo\"],\"timeout_s\":5,\"mem_mb\":64,\"outputs\":[\"../y\"]}",
    ];
    for c in cases {
        let (st, _, _) = http(port, "POST", "/run", Some(c.to_string())).await;
        assert_eq!(st, 400, "case={c}");
    }
    assert_eq!(http(port, "GET", "/health", None).await.0, 200);
}

#[tokio::test]
async fn queue_overflow_503() {
    let port = start(1, 1).await; // 1 worker, queue depth 1
    let slow = run_body(
        vec![],
        vec!["/bin/sleep".into(), "2".into()],
        10,
        256,
        vec![],
    );
    let t1 = tokio::spawn(http(port, "POST", "/run", Some(slow)));
    tokio::time::sleep(Duration::from_millis(300)).await; // worker busy on slow job
    let fast = run_body(
        vec![],
        vec!["/bin/echo".into(), "x".into()],
        10,
        256,
        vec![],
    );
    let (r2, r3) = tokio::join!(
        http(port, "POST", "/run", Some(fast.clone())),
        http(port, "POST", "/run", Some(fast.clone()))
    );
    let (s2, h2, _) = r2;
    let (s3, h3, _) = r3;
    let codes = [s2, s3];
    assert!(
        codes.contains(&200) && codes.contains(&503),
        "expected one 200 and one 503, got {codes:?}"
    );
    let h503 = if s2 == 503 { &h2 } else { &h3 };
    assert_eq!(h503.get("retry-after").map(String::as_str), Some("1"));
    let (st1, _, _) = t1.await.unwrap();
    assert_eq!(st1, 200);
    assert_eq!(http(port, "GET", "/health", None).await.0, 200);
}

/// Value of a Prometheus series (exact `name{labels}` prefix), if present.
fn metric_val(text: &str, series: &str) -> Option<f64> {
    text.lines()
        .find(|l| l.starts_with(series) && !l.starts_with('#'))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
}

#[tokio::test]
async fn metrics_counts_exact() {
    let port = start(2, 8).await;

    // run 1: ok (exit 0)
    let b1 = run_body(vec![], vec!["/bin/echo".into(), "hi".into()], 5, 256, vec![]);
    let (st, _, _) = http(port, "POST", "/run", Some(b1)).await;
    assert_eq!(st, 200);

    // run 2: nonzero (exit 3)
    let b2 = run_body(
        vec![],
        vec!["sh".into(), "-c".into(), "exit 3".into()],
        5,
        256,
        vec![],
    );
    let (st, _, _) = http(port, "POST", "/run", Some(b2)).await;
    assert_eq!(st, 200);

    // run 3: timeout (sleep 30 @ timeout 1)
    let b3 = run_body(vec![], vec!["/bin/sleep".into(), "30".into()], 1, 256, vec![]);
    let (st, _, _) = http(port, "POST", "/run", Some(b3)).await;
    assert_eq!(st, 200);

    let (st, hdrs, text) = http(port, "GET", "/metrics", None).await;
    assert_eq!(st, 200, "metrics endpoint missing");
    assert!(
        hdrs
            .get("content-type")
            .map(|c| c.starts_with("text/plain"))
            .unwrap_or(false),
        "expected text/plain, got {:?}",
        hdrs.get("content-type")
    );

    // exact outcome counts: 1/1/1, none rejected
    assert_eq!(metric_val(&text, r#"lgbox_runs_total{outcome="ok"}"#), Some(1.0));
    assert_eq!(
        metric_val(&text, r#"lgbox_runs_total{outcome="nonzero"}"#),
        Some(1.0)
    );
    assert_eq!(
        metric_val(&text, r#"lgbox_runs_total{outcome="timeout"}"#),
        Some(1.0)
    );
    assert_eq!(
        metric_val(&text, r#"lgbox_runs_total{outcome="rejected"}"#),
        Some(0.0)
    );

    // histogram observed all 3 completed runs
    assert_eq!(metric_val(&text, "lgbox_run_seconds_count"), Some(3.0));

    // gauges drained to 0 now that every response has been received
    assert_eq!(metric_val(&text, "lgbox_queue_depth"), Some(0.0));
    assert_eq!(metric_val(&text, "lgbox_workers_busy"), Some(0.0));
}
