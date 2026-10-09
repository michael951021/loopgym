# lgbox firecracker backend (fake VMM tested)

## Issue
Process rlimits are weak isolation: a crafted child can exploit kernel bugs or
abuse shared-uid NPROC pressure. Production RL grading (tens of thousands of
untrusted programs) needs microVM isolation. Firecracker boots a VM from a
snapshot in ~0.6 s on this host, so a pool of pre-booted VMs can grade with
near-process latency.

## Question
Does a `VmPool` of `Vmm` backends lease round-robin, reset after each run,
replace a crashed VM on the next lease, and queue (not drop) when all VMs are
busy? Refutation: with 2 VMs, sequential runs do not alternate 0,1,0,1; a
crashed VM is re-executed without re-boot; with 1 VM, a concurrent second run
is rejected/lost instead of running after the first; a used VM is reused
without a reset.

## Evidence
- lgbox/src/lib.rs: process backend only; no Vmm trait (observed).
- Firecracker API field names checked against firecracker main branch source
  (machine_config/boot_source/drive/vsock/vmm_config + snapshot request
  parser): vcpu_count, mem_size_mib, kernel_image_path, boot_args,
  drive_id/path_on_host/is_root_device, guest_cid/uds_path,
  snapshot/load {snapshot_path, mem_backend{backend_path,backend_type:"File"},
  resume_vm}, /actions {"action_type":"InstanceStart"}.
- New tests: lgbox/tests/firecracker.rs (FakeVmm only — no /dev/kvm in
  sandbox, rule 4). Evidence: .agent-evidence/lgboxfc-*.log.

## Done when
1. `cd lgbox && PATH=$PWD/../.tools/cargo/bin:$PATH cargo test` passes, incl.
   lease/reset ordering (2 VMs alternate; boot only on first use; reset after
   every run), crashed-VM replacement (dead slot re-boots, run succeeds),
   pool-exhaustion queueing (1 VM, 2 concurrent runs both succeed, second
   starts after first ends).
2. `lgbox serve --backend firecracker --dry-run` exits 0 and prints a JSON
   config with machine-config/boot-source/drives/vsock/snapshot-load fields
   that validates (non-empty paths, vcpu>=1, mem>=1, guest_cid>=3).
   Known-answer check: the printed JSON must parse and carry exactly the
   flag values passed in (hand-verified fields), and bad flags → nonzero exit.
3. docs/firecracker.md: rootfs + agent protocol + host run steps.
Regression: existing integration tests (process backend) still pass.

## Stop when
Scope: Rust side only (trait, pool, real-socket impl, config, docs). No
snapshot *creation* tooling, no /dev/kvm test, no guest-agent build here, no
vLLM/Python changes. Real VMM is untestable in sandbox (no kvm, no
firecracker binary) — it must compile and be plausible, exercised only by
FakeVmm in tests. Two attempts without new evidence on a failing pool test
→ record in NOTES.md, stop or split.

## Plan
1. src/vm.rs: Vmm trait, ExecOut, VmPool (Mutex+Condvar, round-robin lease,
   lazy re-boot of dead slots, run_job = boot?→put→exec→get→reset), preboot.
2. src/fc.rs: VmConfig + validate + api JSON; FirecrackerVmm over the
   firecracker API unix socket (std UnixStream HTTP) + guest agent over the
   vsock UDS (JSON lines).
3. lib.rs: Runner trait; app takes runner; ProcessRunner default (keep
   app(workers,queue)); VmPool implements Runner.
4. main.rs: --backend, --vm-size, --kernel/--rootfs/--vcpu/--mem-mib/
   --snapshot/--mem-file, --dry-run (print config, exit 0).
5. tests/firecracker.rs with FakeVmm; docs/firecracker.md.
