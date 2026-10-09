//! Firecracker backend: VM config + a `Vmm` that drives one firecracker
//! instance over its API unix socket, and a guest agent over the vsock UDS.
//!
//! API field names follow the firecracker main branch (vmm_config
//! machine_config/boot_source/drive/vsock, snapshot request parser):
//! machine-config {vcpu_count, mem_size_mib}, boot-source {kernel_image_path,
//! boot_args}, drives {drive_id, path_on_host, is_root_device}, vsock
//! {guest_cid, uds_path}, snapshot/load {snapshot_path,
//! mem_backend{backend_path, backend_type:"File"}, resume_vm}.
//!
//! The guest agent listens on vsock port 52 and speaks JSON lines:
//! {"op":"put","files":{path:content}}      -> {"ok":true}
//! {"op":"exec","cmd":[...],"timeout_s":N}  -> {"exit_code":N,"timed_out":b,
//!                                              "stdout":"...","stderr":"..."}
//! {"op":"get","paths":[...]}               -> {"files":{path:content|null}}
//! {"op":"reset"}                           -> {"ok":true}
//! See docs/firecracker.md for the rootfs build.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::vm::{ExecOut, Vmm};

#[derive(Clone, Debug)]
pub struct VmConfig {
    pub kernel: String,
    pub rootfs: String,
    pub snapshot: String,
    pub mem_file: String,
    pub vcpu: u8,
    pub mem_mib: u32,
    pub guest_cid: u32,
    pub uds_path: String,
    pub bin: String,
}

impl VmConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("kernel", &self.kernel),
            ("rootfs", &self.rootfs),
            ("snapshot", &self.snapshot),
            ("mem_file", &self.mem_file),
            ("uds_path", &self.uds_path),
            ("bin", &self.bin),
        ] {
            if v.is_empty() {
                return Err(format!("{name} must be non-empty"));
            }
        }
        if self.vcpu == 0 {
            return Err("vcpu must be >= 1".into());
        }
        if self.mem_mib == 0 {
            return Err("mem_mib must be >= 1".into());
        }
        if self.guest_cid < 3 {
            return Err("guest_cid must be >= 3".into());
        }
        Ok(())
    }

    /// The VM config as firecracker API bodies: the boot-time endpoints (for
    /// the source VM the snapshot is taken from) plus the snapshot/load body
    /// used to restore each pool VM.
    pub fn api_config(&self) -> Value {
        json!({
            "machine-config": {
                "vcpu_count": self.vcpu,
                "mem_size_mib": self.mem_mib,
                "smt": false
            },
            "boot-source": {
                "kernel_image_path": self.kernel,
                "boot_args": "console=ttyS0 reboot=k panic=1 pci=off"
            },
            "drives": [{
                "drive_id": "rootfs",
                "path_on_host": self.rootfs,
                "is_root_device": true,
                "is_read_only": false
            }],
            "vsock": {
                "guest_cid": self.guest_cid,
                "uds_path": self.uds_path
            },
            "snapshot/load": {
                "snapshot_path": self.snapshot,
                "mem_backend": {
                    "backend_path": self.mem_file,
                    "backend_type": "File"
                },
                "resume_vm": true
            }
        })
    }
}

pub struct FirecrackerVmm {
    cfg: VmConfig,
    id: u32,
    sock: PathBuf,
    proc: Mutex<Option<Child>>,
}

impl FirecrackerVmm {
    pub fn new(cfg: VmConfig, id: u32) -> Self {
        let sock = PathBuf::from(format!("{}-api", cfg.uds_path));
        Self { cfg, id, sock, proc: Mutex::new(None) }
    }

    /// One HTTP request against the firecracker API unix socket.
    fn api(&self, method: &str, path: &str, body: Option<&Value>) -> Result<(), String> {
        let mut st = UnixStream::connect(&self.sock).map_err(|e| format!("api socket: {e}"))?;
        let payload = body.map(|b| b.to_string()).unwrap_or_default();
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        st.write_all(req.as_bytes()).map_err(|e| format!("api write: {e}"))?;
        let mut resp = Vec::new();
        let mut buf = [0u8; 4096];
        while let Ok(n) = st.read(&mut buf) {
            if n == 0 {
                break;
            }
            resp.extend_from_slice(&buf[..n]);
        }
        let text = String::from_utf8_lossy(&resp);
        let status = text
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|c| c.parse::<u16>().ok())
            .ok_or_else(|| format!("bad api reply: {text}"))?;
        if (200..300).contains(&status) {
            Ok(())
        } else {
            Err(format!("api {method} {path} -> {status}: {text}"))
        }
    }

    /// One JSON-line request to the guest agent over the vsock UDS.
    fn agent(&self, op: &Value, deadline: Instant) -> Result<Value, String> {
        let mut st = UnixStream::connect(&self.cfg.uds_path).map_err(|e| format!("vsock: {e}"))?;
        st.set_write_timeout(Some(Duration::from_secs(10))).ok();
        st.set_read_timeout(Some(Duration::from_secs(1))).ok();
        st.write_all((op.to_string() + "\n").as_bytes())
            .map_err(|e| format!("vsock write: {e}"))?;
        let mut resp = Vec::new();
        let mut buf = [0u8; 65536];
        loop {
            match st.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    resp.extend_from_slice(&buf[..n]);
                    if resp.contains(&b'\n') {
                        break;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(format!("agent timeout: {}", op["op"]));
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(format!("vsock read: {e}")),
            }
        }
        serde_json::from_str(String::from_utf8_lossy(&resp).trim())
            .map_err(|e| format!("agent reply: {e}"))
    }

    fn check_ok(v: &Value) -> Result<(), String> {
        if v.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(())
        } else {
            Err(format!("agent error: {v}"))
        }
    }

    fn boot(&self) -> Result<(), String> {
        if let Some(mut c) = self.proc.lock().unwrap().take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        let _ = std::fs::remove_file(&self.sock);
        let child = Command::new(&self.cfg.bin)
            .arg("--api-sock")
            .arg(&self.sock)
            .arg("--id")
            .arg(self.id.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", self.cfg.bin))?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if UnixStream::connect(&self.sock).is_ok() {
                break;
            }
            if Instant::now() >= deadline {
                return Err("firecracker api socket did not appear".into());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let body = json!({
            "snapshot_path": self.cfg.snapshot,
            "mem_backend": {
                "backend_path": self.cfg.mem_file,
                "backend_type": "File"
            },
            "resume_vm": true
        });
        self.api("PUT", "/snapshot/load", Some(&body))?;
        *self.proc.lock().unwrap() = Some(child);
        Ok(())
    }
}

impl Vmm for FirecrackerVmm {
    fn boot_from_snapshot(&self) -> Result<(), String> {
        self.boot()
    }

    fn put_files(&self, files: &BTreeMap<String, String>) -> Result<(), String> {
        Self::check_ok(&self.agent(&json!({"op": "put", "files": files}),
            Instant::now() + Duration::from_secs(30))?)
    }

    fn exec(&self, cmd: &[String], timeout_s: u32) -> Result<ExecOut, String> {
        let deadline = Instant::now() + Duration::from_secs(timeout_s as u64) + Duration::from_secs(10);
        let v = self.agent(&json!({"op": "exec", "cmd": cmd, "timeout_s": timeout_s}), deadline)?;
        Ok(ExecOut {
            exit_code: v["exit_code"].as_i64().unwrap_or(-1) as i32,
            timed_out: v["timed_out"].as_bool().unwrap_or(false),
            stdout: v["stdout"].as_str().unwrap_or_default().into(),
            stderr: v["stderr"].as_str().unwrap_or_default().into(),
        })
    }

    fn get_files(&self, paths: &[String]) -> Result<BTreeMap<String, Option<String>>, String> {
        let v = self.agent(&json!({"op": "get", "paths": paths}), Instant::now() + Duration::from_secs(30))?;
        let files = v
            .get("files")
            .and_then(Value::as_object)
            .ok_or_else(|| "get: no files in reply".to_string())?;
        Ok(paths
            .iter()
            .map(|p| (p.clone(), files.get(p).and_then(Value::as_str).map(str::to_string)))
            .collect())
    }

    fn reset(&self) -> Result<(), String> {
        Self::check_ok(&self.agent(&json!({"op": "reset"}), Instant::now() + Duration::from_secs(30))?)
    }
}

impl Drop for FirecrackerVmm {
    fn drop(&mut self) {
        if let Some(mut c) = self.proc.lock().unwrap().take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}
