use std::net::SocketAddr;
use std::sync::Arc;

use lgbox::fc::{FirecrackerVmm, VmConfig};
use lgbox::vm::{VmPool, Vmm};
use lgbox::Runner;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("serve") {
        usage();
    }
    let mut port: u16 = 7777;
    let mut workers: usize = 4;
    let mut queue: usize = 64;
    let mut backend = "process".to_string();
    let mut vm_size: Option<usize> = None;
    let mut dry = false;
    let mut cfg = VmConfig {
        kernel: "/var/lib/lgbox/vmlinux".into(),
        rootfs: "/var/lib/lgbox/rootfs.ext4".into(),
        snapshot: "/var/lib/lgbox/snapshot.vmstate".into(),
        mem_file: "/var/lib/lgbox/snapshot.mem".into(),
        vcpu: 1,
        mem_mib: 512,
        guest_cid: 3,
        uds_path: "/tmp/lgbox-fc-3.sock".into(),
        bin: std::env::var("LGBOX_FIRECRACKER_BIN").unwrap_or_else(|_| "firecracker".into()),
    };
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--dry-run" {
            dry = true;
            i += 1;
            continue;
        }
        let v = match args.get(i + 1) {
            Some(v) => v.as_str(),
            None => die(&format!("missing value for {}", args[i])),
        };
        match args[i].as_str() {
            "--port" => port = parse(v, "port"),
            "--workers" => workers = parse(v, "workers"),
            "--queue" => queue = parse(v, "queue"),
            "--backend" => backend = v.to_string(),
            "--vm-size" => vm_size = Some(parse(v, "vm-size")),
            "--kernel" => cfg.kernel = v.to_string(),
            "--rootfs" => cfg.rootfs = v.to_string(),
            "--snapshot" => cfg.snapshot = v.to_string(),
            "--mem-file" => cfg.mem_file = v.to_string(),
            "--vcpu" => cfg.vcpu = parse(v, "vcpu"),
            "--mem-mib" => cfg.mem_mib = parse(v, "mem-mib"),
            "--uds" => cfg.uds_path = v.to_string(),
            "--fc-bin" => cfg.bin = v.to_string(),
            other => die(&format!("unknown option {other}")),
        }
        i += 2;
    }
    if workers == 0 || queue == 0 {
        die("workers and queue must be >= 1");
    }
    match backend.as_str() {
        "process" => {}
        "firecracker" => {}
        other => die(&format!("backend must be process or firecracker, got {other}")),
    }
    if dry {
        if backend != "firecracker" {
            die("--dry-run requires --backend firecracker");
        }
        if let Err(e) = cfg.validate() {
            die(&e);
        }
        println!("{}", serde_json::to_string_pretty(&cfg.api_config()).unwrap());
        return;
    }
    let runner: Arc<dyn Runner> = if backend == "firecracker" {
        if let Err(e) = cfg.validate() {
            die(&e);
        }
        let n = vm_size.unwrap_or(workers).max(1);
        let vms: Vec<Arc<dyn Vmm>> = (0..n)
            .map(|i| {
                let mut c = cfg.clone();
                c.guest_cid = cfg.guest_cid + i as u32;
                c.uds_path = format!("/tmp/lgbox-fc-{}.sock", c.guest_cid);
                let id = c.guest_cid;
                let vm: Arc<dyn Vmm> = Arc::new(FirecrackerVmm::new(c, id));
                vm
            })
            .collect();
        let pool = Arc::new(VmPool::new(vms));
        std::thread::spawn({
            let p = pool.clone();
            move || println!("lgbox prebooted {}/{} firecracker vms", p.preboot(), p.size())
        });
        pool
    } else {
        Arc::new(lgbox::ProcessRunner)
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        let addr: SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        println!(
            "lgbox listening on {} workers={workers} queue={queue} backend={backend}",
            listener.local_addr().unwrap()
        );
        axum::serve(listener, lgbox::app_with_runner(workers, queue, runner)).await.unwrap();
    });
}

fn usage() -> ! {
    eprintln!(
        "usage: lgbox serve --port P --workers N --queue Q \
         [--backend process|firecracker] [--vm-size N] \
         [--kernel P] [--rootfs P] [--snapshot P] [--mem-file P] \
         [--vcpu N] [--mem-mib N] [--uds P] [--fc-bin P] [--dry-run]"
    );
    std::process::exit(2);
}

fn parse<T: std::str::FromStr>(s: &str, what: &str) -> T
where
    T::Err: std::fmt::Display,
{
    s.parse().unwrap_or_else(|e: T::Err| die(&format!("{what}: {e}")))
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}
