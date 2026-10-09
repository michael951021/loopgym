//! VmPool behavior with a FakeVmm (no /dev/kvm in the sandbox, rule 4):
//! lease/reset ordering, crashed-VM replacement, pool-exhaustion queueing.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lgbox::fc::VmConfig;
use lgbox::vm::{ExecOut, Vmm, VmPool};
use lgbox::RunReq;

fn req() -> RunReq {
    RunReq {
        files: BTreeMap::new(),
        cmd: vec!["true".into()],
        timeout_s: 10,
        mem_mb: 256,
        outputs: vec![],
    }
}

struct Inner {
    execs: usize,
    boots: usize,
    die_after: Option<usize>,
    fail_boots: usize,
    fail_resets: Option<usize>,
    sleep_ms: u64,
}

struct Fake {
    id: usize,
    ops: Arc<Mutex<Vec<(usize, &'static str)>>>,
    windows: Arc<Mutex<Vec<(usize, Instant, Instant)>>>,
    inner: Mutex<Inner>,
}

impl Fake {
    fn new(id: usize, ops: Arc<Mutex<Vec<(usize, &'static str)>>>, windows: Arc<Mutex<Vec<(usize, Instant, Instant)>>>) -> Self {
        Self {
            id,
            ops,
            windows,
            inner: Mutex::new(Inner {
                execs: 0,
                boots: 0,
                die_after: None,
                fail_boots: 0,
                fail_resets: None,
                sleep_ms: 0,
            }),
        }
    }

    fn set(&self, f: impl FnOnce(&mut Inner)) {
        f(&mut self.inner.lock().unwrap());
    }

}

impl Vmm for Fake {
    fn boot_from_snapshot(&self) -> Result<(), String> {
        let mut g = self.inner.lock().unwrap();
        g.boots += 1;
        if g.fail_boots > 0 {
            g.fail_boots -= 1;
            return Err("no /dev/kvm".into());
        }
        drop(g);
        self.ops.lock().unwrap().push((self.id, "boot"));
        Ok(())
    }

    fn put_files(&self, _: &BTreeMap<String, String>) -> Result<(), String> {
        self.ops.lock().unwrap().push((self.id, "put"));
        Ok(())
    }

    fn exec(&self, _: &[String], _: u32) -> Result<ExecOut, String> {
        let mut g = self.inner.lock().unwrap();
        g.execs += 1;
        let n = g.execs;
        let sleep_ms = g.sleep_ms;
        drop(g);
        self.ops.lock().unwrap().push((self.id, "exec"));
        let t0 = Instant::now();
        if sleep_ms > 0 {
            std::thread::sleep(Duration::from_millis(sleep_ms));
        }
        let t1 = Instant::now();
        self.windows.lock().unwrap().push((self.id, t0, t1));
        let g = self.inner.lock().unwrap();
        if g.die_after == Some(n) {
            return Err("guest died".into());
        }
        Ok(ExecOut {
            exit_code: 0,
            timed_out: false,
            stdout: format!("vm{} out", self.id),
            stderr: String::new(),
        })
    }

    fn get_files(&self, paths: &[String]) -> Result<BTreeMap<String, Option<String>>, String> {
        self.ops.lock().unwrap().push((self.id, "get"));
        Ok(paths.iter().map(|p| (p.clone(), Some("c".into()))).collect())
    }

    fn reset(&self) -> Result<(), String> {
        self.ops.lock().unwrap().push((self.id, "reset"));
        let mut g = self.inner.lock().unwrap();
        if g.fail_resets.is_some() {
            g.fail_resets = None;
            return Err("agent gone".into());
        }
        Ok(())
    }
}

#[test]
fn lease_round_robin_and_reset() {
    let ops = Arc::new(Mutex::new(Vec::new()));
    let wins = Arc::new(Mutex::new(Vec::new()));
    let pool = VmPool::new(vec![
        Arc::new(Fake::new(0, ops.clone(), wins.clone())),
        Arc::new(Fake::new(1, ops.clone(), wins.clone())),
    ]);
    for _ in 0..4 {
        assert_eq!(pool.run_job(&req()).exit_code, Some(0));
    }
    let all = ops.lock().unwrap().clone();
    let execs: Vec<usize> = all.iter().filter(|o| o.1 == "exec").map(|o| o.0).collect();
    assert_eq!(execs, vec![0, 1, 0, 1], "leases must alternate round-robin");
    for id in [0usize, 1] {
        let seq: Vec<&str> = all.iter().filter(|o| o.0 == id).map(|o| o.1).collect();
        assert_eq!(
            seq,
            vec!["boot", "put", "exec", "get", "reset", "put", "exec", "get", "reset"],
            "vm {id}: boot once, reset after every run, no re-boot"
        );
    }
}

#[test]
fn crashed_vm_is_replaced() {
    let ops = Arc::new(Mutex::new(Vec::new()));
    let wins = Arc::new(Mutex::new(Vec::new()));
    let f = Fake::new(0, ops.clone(), wins);
    f.set(|i| i.die_after = Some(2)); // second exec (run 2) crashes the guest
    let pool = VmPool::new(vec![Arc::new(f)]);
    assert_eq!(pool.run_job(&req()).exit_code, Some(0));
    let r2 = pool.run_job(&req());
    assert_eq!(r2.exit_code, None, "crashed run must fail");
    assert!(r2.stderr_tail.contains("exec:"), "stderr: {}", r2.stderr_tail);
    assert_eq!(pool.run_job(&req()).exit_code, Some(0), "replacement vm must work");
    let seq: Vec<&str> = ops.lock().unwrap().iter().map(|o| o.1).collect();
    assert_eq!(
        seq,
        vec![
            "boot", "put", "exec", "get", "reset", // run 1
            "put", "exec",                          // run 2 crashes mid-exec
            "boot", "put", "exec", "get", "reset",  // run 3 re-boots
        ],
        "a crashed vm is re-booted on the next lease"
    );
}

#[test]
fn pool_exhaustion_queues() {
    let ops = Arc::new(Mutex::new(Vec::new()));
    let wins = Arc::new(Mutex::new(Vec::new()));
    let f = Fake::new(0, ops.clone(), wins.clone());
    f.set(|i| i.sleep_ms = 150);
    let pool = Arc::new(VmPool::new(vec![Arc::new(f)]));
    let p1 = pool.clone();
    let t1 = std::thread::spawn(move || p1.run_job(&req()));
    let p2 = pool.clone();
    let t2 = std::thread::spawn(move || p2.run_job(&req()));
    let o1 = t1.join().unwrap();
    let o2 = t2.join().unwrap();
    assert_eq!(o1.exit_code, Some(0));
    assert_eq!(o2.exit_code, Some(0), "second run must queue, not be dropped");
    let w = wins.lock().unwrap();
    assert_eq!(w.len(), 2);
    assert!(w[1].1 >= w[0].2, "single-vm pool must serialize execs");
}

#[test]
fn reset_failure_marks_dead_but_keeps_result() {
    let ops = Arc::new(Mutex::new(Vec::new()));
    let wins = Arc::new(Mutex::new(Vec::new()));
    let f = Fake::new(0, ops.clone(), wins);
    f.set(|i| i.fail_resets = Some(1));
    let pool = VmPool::new(vec![Arc::new(f)]);
    assert_eq!(pool.run_job(&req()).exit_code, Some(0), "result kept");
    assert_eq!(pool.run_job(&req()).exit_code, Some(0));
    let seq: Vec<&str> = ops.lock().unwrap().iter().map(|o| o.1).collect();
    assert_eq!(
        seq,
        vec![
            "boot", "put", "exec", "get", "reset", // run 1, reset fails
            "boot", "put", "exec", "get", "reset",  // run 2 re-boots
        ],
    );
}

#[test]
fn boot_failure_retried_on_next_lease() {
    let ops = Arc::new(Mutex::new(Vec::new()));
    let wins = Arc::new(Mutex::new(Vec::new()));
    let f = Fake::new(0, ops.clone(), wins);
    f.set(|i| i.fail_boots = 1);
    let pool = VmPool::new(vec![Arc::new(f)]);
    let r1 = pool.run_job(&req());
    assert_eq!(r1.exit_code, None);
    assert!(r1.stderr_tail.contains("boot:"), "stderr: {}", r1.stderr_tail);
    assert_eq!(pool.run_job(&req()).exit_code, Some(0));
}

#[test]
fn preboot_boots_all_slots() {
    let ops = Arc::new(Mutex::new(Vec::new()));
    let wins = Arc::new(Mutex::new(Vec::new()));
    let f0 = Fake::new(0, ops.clone(), wins.clone());
    let f1 = Fake::new(1, ops.clone(), wins.clone());
    let pool = VmPool::new(vec![Arc::new(f0), Arc::new(f1)]);
    assert_eq!(pool.preboot(), 2);
    assert_eq!(pool.run_job(&req()).exit_code, Some(0));
    let all = ops.lock().unwrap().clone();
    let boots = all.iter().filter(|o| o.1 == "boot").count();
    assert_eq!(boots, 2, "pre-booted vms must not re-boot on first run");
}

fn cfg() -> VmConfig {
    VmConfig {
        kernel: "/k/vmlinux".into(),
        rootfs: "/k/rootfs.ext4".into(),
        snapshot: "/k/s.vmstate".into(),
        mem_file: "/k/s.mem".into(),
        vcpu: 1,
        mem_mib: 512,
        guest_cid: 3,
        uds_path: "/tmp/lgbox-fc-3.sock".into(),
        bin: "firecracker".into(),
    }
}

#[test]
fn config_validation() {
    assert!(cfg().validate().is_ok());
    let mut c = cfg();
    c.vcpu = 0;
    assert!(c.validate().is_err());
    let mut c = cfg();
    c.kernel = String::new();
    assert!(c.validate().is_err());
    let mut c = cfg();
    c.guest_cid = 2;
    assert!(c.validate().is_err());
    let mut c = cfg();
    c.mem_mib = 0;
    assert!(c.validate().is_err());
    let mut c = cfg();
    c.snapshot = String::new();
    assert!(c.validate().is_err());
}

#[test]
fn api_config_fields() {
    let v = cfg().api_config();
    assert_eq!(v["machine-config"]["vcpu_count"], 1);
    assert_eq!(v["machine-config"]["mem_size_mib"], 512);
    assert_eq!(v["boot-source"]["kernel_image_path"], "/k/vmlinux");
    assert!(v["boot-source"]["boot_args"].as_str().unwrap().contains("console=ttyS0"));
    assert_eq!(v["drives"][0]["drive_id"], "rootfs");
    assert_eq!(v["drives"][0]["path_on_host"], "/k/rootfs.ext4");
    assert_eq!(v["drives"][0]["is_root_device"], true);
    assert_eq!(v["vsock"]["guest_cid"], 3);
    assert_eq!(v["vsock"]["uds_path"], "/tmp/lgbox-fc-3.sock");
    assert_eq!(v["snapshot/load"]["snapshot_path"], "/k/s.vmstate");
    assert_eq!(v["snapshot/load"]["mem_backend"]["backend_path"], "/k/s.mem");
    assert_eq!(v["snapshot/load"]["mem_backend"]["backend_type"], "File");
    assert_eq!(v["snapshot/load"]["resume_vm"], true);
}
