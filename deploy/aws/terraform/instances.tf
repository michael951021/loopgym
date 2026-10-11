# Spot worker instances for the scaling study (ticket `aws` supplies the
# training bootstrap). Every instance is capped from outside itself:
#   - user_data shuts the box down at max_hours, FIRST, before anything that
#     can fail (a hung bootstrap cannot stop the shutdown)
#   - one-time spot, terminate on interruption and on shutdown
#   - gp3 root, delete_on_termination, no extra volumes, no EIP
#   - IMDSv2 required (no metadata via IMDSv1)
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
  user_data_replace_on_change = true

  # Line 1 is the hard stop, before anything that can fail.
  user_data = <<-EOF
    #!/bin/bash
    shutdown -h +$(( ${var.max_hours} * 60 ))
    echo "loopgym run ${var.run_id}: worker ${count.index} up (study bootstrap: ticket aws)"
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
