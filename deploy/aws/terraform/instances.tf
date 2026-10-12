# Spot worker instances for the scaling study (ticket `aws`). Every instance
# is capped from outside itself:
#   - user_data shuts the box down at max_hours, FIRST, before anything that
#     can fail (a hung bootstrap cannot stop the shutdown)
#   - one-time spot, terminate on interruption and on shutdown
#   - gp3 root, delete_on_termination, no extra volumes, no EIP
#   - IMDSv2 required (no metadata via IMDSv1)
#
# The bootstrap after the shutdown line installs the NVIDIA driver (which
# reboots once) and builds .venv-gpu. cloud-init re-runs user_data from the
# top after that reboot, so every step is idempotent and the shutdown timer
# is re-armed before and after the reboot. The out-of-instance scheduler is
# the absolute cap either way.
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_instance" "worker" {
  count = var.nodes

  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.run.id
  vpc_security_group_ids = [aws_security_group.workers.id]
  placement_group        = aws_placement_group.workers.id
  key_name               = aws_key_pair.workers.key_name
  associate_public_ip_address = true
  # Node 0 is the c10d master (MASTER_IP_AWS in loopgym/rl/launch.py).
  private_ip = count.index == 0 ? "10.0.0.10" : "10.0.0.11"
  user_data_replace_on_change = true

  user_data = <<-EOF
    #!/bin/bash
    shutdown -h +$(( ${var.max_hours} * 60 ))
    # Study bootstrap, AFTER the hard stop. Idempotent: cloud-init re-runs this
    # from the top after the driver-install reboot, so each step guards itself
    # and the shutdown timer is re-armed around the reboot.
    set -euo pipefail
    exec >> /var/log/loopgym-bootstrap.log 2>&1
    echo "loopgym run ${var.run_id}: worker ${count.index} bootstrapping"
    export DEBIAN_FRONTEND=noninteractive
    if ! command -v nvidia-smi >/dev/null; then
      apt-get update -y
      apt-get install -y git python3.12 python3.12-venv
      for v in 580 575 570 565 560 550 535; do
        if apt-get install -y "nvidia-driver-$v"; then break; fi
      done
      command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver installed"; exit 1; }
      # Re-arm the hard stop across the reboot, then reboot.
      shutdown -h +$(( ${var.max_hours} * 60 ))
      reboot; exit 0
    fi
    # Resumed from the driver-install reboot: re-arm, verify the GPU.
    shutdown -h +$(( ${var.max_hours} * 60 ))
    nvidia-smi || exit 1
    [ -d /work/.git ] || git clone ${var.repo_url} /work
    cd /work
    [ -x .venv-gpu/bin/python ] || python3.12 -m venv .venv-gpu
    .venv-gpu/bin/python -m pip install -q uv
    vllm_pin=$(sed -n 's/^vllm=//p' tools/versions.txt | head -n 1)
    ds_pin=$(sed -n 's/^deepspeed=//p' tools/versions.txt | head -n 1)
    [ -n "$vllm_pin" ] && [ -n "$ds_pin" ] || { echo "tools/versions.txt must pin vllm= and deepspeed="; exit 1; }
    .venv-gpu/bin/uv pip install -q --python .venv-gpu/bin/python \
      "torch" "vllm==$vllm_pin" "deepspeed==$ds_pin" "triton" "ninja" \
      "transformers" "peft" "accelerate" "safetensors" "numpy" \
      "pydantic" "httpx" "tomli-w" "pytest"
    .venv-gpu/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch has no CUDA"
print("torch", torch.__version__, "GPUs", torch.cuda.device_count())
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-Coder-0.5B-Instruct",
                  local_dir="models/qwen2.5-coder-0.5b-instruct")
PY
    touch /work/.aws-ready
    echo "loopgym run ${var.run_id}: worker ${count.index} ready"
  EOF

  # One-time spot: a single bid, terminate when the price is beaten.
  instance_market_options {
    market_type = "spot"

    spot_options {
      instance_interruption_behavior = "terminate"
      max_price                      = var.max_price
      spot_instance_type             = "one-time"
    }
  }

  # Terminate (not stop) when we ourselves ask it to end.
  instance_initiated_shutdown_behavior = "terminate"

  # IMDSv2 required: no insecure IMDSv1 metadata access.
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  # gp3 root that dies with the instance; no extra volumes.
  root_block_device {
    volume_size            = 50
    volume_type            = "gp3"
    delete_on_termination  = true
  }

  tags = {
    Role = "worker-${count.index}"
  }
}
