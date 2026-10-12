# Per-run network (ticket `aws`): a throwaway VPC that dies with
# `terraform destroy`. No NAT gateway, load balancer, EIP or S3 — anything
# that bills while idle is banned (rule 4). SSH comes from var.admin_cidr
# only; the rest of the world is closed.
resource "aws_vpc" "run" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags                 = { Name = "loopgym-${var.run_id}" }
}

resource "aws_internet_gateway" "run" {
  vpc_id = aws_vpc.run.id
  tags   = { Name = "loopgym-${var.run_id}" }
}

resource "aws_subnet" "run" {
  vpc_id            = aws_vpc.run.id
  cidr_block        = "10.0.0.0/24"
  availability_zone = "${var.region}a"
  tags              = { Name = "loopgym-${var.run_id}" }
}

resource "aws_route_table" "run" {
  vpc_id = aws_vpc.run.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.run.id
  }
}

resource "aws_route_table_association" "run" {
  subnet_id      = aws_subnet.run.id
  route_table_id = aws_route_table.run.id
}

resource "aws_security_group" "workers" {
  name        = "loopgym-${var.run_id}"
  description = "Scaling study workers: SSH from the admin host, everything in-run."
  vpc_id      = aws_vpc.run.id

  ingress {
    description = "SSH from the admin host only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr]
  }

  ingress {
    description = "run-internal (torchrun rendezvous, NCCL)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Cluster placement: the two workers land in the same AZ cell of the subnet.
resource "aws_placement_group" "workers" {
  name     = "loopgym-${var.run_id}"
  strategy = "cluster"
}

# SSH key for this run only; destroyed with the VPC.
resource "aws_key_pair" "workers" {
  key_name   = "loopgym-${var.run_id}"
  public_key = file(pathexpand(var.ssh_public_key_path))
}
