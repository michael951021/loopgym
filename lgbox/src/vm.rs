//! MicroVM backend: a `Vmm` trait per VM and a `VmPool` that leases them
//! round-robin, re-boots crashed ones, and queues when the pool is full.

use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::{fail, tail, RunOut, RunReq, Runner};

pub struct ExecOut {
    pub exit_code: i32,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
}

/// One VM. `boot_from_snapshot` (re)starts the VM from the pristine snapshot;
/// `reset` restores the workdir between runs without a re-boot. All methods
/// report a dead VM as `Err`; the pool then re-boots it on the next lease.
pub trait Vmm: Send + Sync {
    fn boot_from_snapshot(&self) -> Result<(), String>;
    fn put_files(&self, files: &BTreeMap<String, String>) -> Result<(), String>;
    fn exec(&self, cmd: &[String], timeout_s: u32) -> Result<ExecOut, String>;
    fn get_files(&self, paths: &[String]) -> Result<BTreeMap<String, Option<String>>, String>;
    fn reset(&self) -> Result<(), String>;
}

#[derive(Default)]
struct Inner {
    next: usize,
    busy: Vec<bool>,
    live: Vec<bool>,
}

pub struct VmPool {
    vms: Vec<Arc<dyn Vmm>>,
    st: Mutex<Inner>,
    cv: Condvar,
}

impl VmPool {
    pub fn new(vms: Vec<Arc<dyn Vmm>>) -> Self {
        assert!(!vms.is_empty(), "pool needs at least one vm");
        let n = vms.len();
        let st = Inner { next: 0, busy: vec![false; n], live: vec![false; n] };
        Self { vms, st: Mutex::new(st), cv: Condvar::new() }
    }

    pub fn size(&self) -> usize {
        self.vms.len()
    }

    /// Boot every dead slot. Returns the number of live slots.
    pub fn preboot(&self) -> usize {
        for i in 0..self.vms.len() {
            if !self.is_live(i) && self.vms[i].boot_from_snapshot().is_ok() {
                self.set_live(i, true);
            }
        }
        self.live_count()
    }

    fn is_live(&self, i: usize) -> bool {
        self.st.lock().unwrap().live[i]
    }

    fn set_live(&self, i: usize, v: bool) {
        self.st.lock().unwrap().live[i] = v;
    }

    fn live_count(&self) -> usize {
        self.st.lock().unwrap().live.iter().filter(|b| **b).count()
    }

    /// Claim the next free slot round-robin from `next`; block until one is
    /// free.
    fn lease(&self) -> usize {
        let g = self.st.lock().unwrap();
        let mut g = self.cv.wait_while(g, |g| g.busy.iter().all(|b| *b)).unwrap();
        let n = g.busy.len();
        let mut i = g.next % n;
        while g.busy[i] {
            i = (i + 1) % n;
        }
        g.busy[i] = true;
        g.next = (i + 1) % n;
        i
    }

    fn release(&self, i: usize) {
        let mut g = self.st.lock().unwrap();
        g.busy[i] = false;
        self.cv.notify_all();
    }

    /// Blocking: lease, boot if dead, put files, exec, get files, reset.
    /// A failure on the VM marks the slot dead (re-booted on next lease);
    /// a reset failure keeps the result but also marks it dead.
    pub fn run_job(&self, req: &RunReq) -> RunOut {
        let start = Instant::now();
        let idx = self.lease();
        let vm = self.vms[idx].clone();
        let out = if self.is_live(idx) {
            Ok(())
        } else {
            vm.boot_from_snapshot().map_err(|e| format!("boot: {e}"))
        }
        .and_then(|_| {
            self.set_live(idx, true);
            vm.put_files(&req.files).map_err(|e| format!("put_files: {e}"))
        })
        .and_then(|_| vm.exec(&req.cmd, req.timeout_s).map_err(|e| format!("exec: {e}")))
        .and_then(|exec| {
            vm.get_files(&req.outputs)
                .map(|outputs| (exec, outputs))
                .map_err(|e| format!("get_files: {e}"))
        });
        match out {
            Ok((exec, outputs)) => {
                if vm.reset().is_err() {
                    self.set_live(idx, false);
                }
                self.release(idx);
                RunOut {
                    exit_code: Some(exec.exit_code),
                    timed_out: exec.timed_out,
                    stdout_tail: tail(&exec.stdout),
                    stderr_tail: tail(&exec.stderr),
                    wall_ms: start.elapsed().as_millis() as u64,
                    outputs,
                }
            }
            Err(e) => {
                self.set_live(idx, false);
                self.release(idx);
                fail(e, start)
            }
        }
    }
}

impl Runner for VmPool {
    fn run(&self, req: &RunReq) -> RunOut {
        self.run_job(req)
    }
}
