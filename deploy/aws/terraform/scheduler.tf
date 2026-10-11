# Out-of-instance kill switch: a one-time schedule that fires
# max_hours + 15 min after launch and terminates the run's instances via a
# universal target, even if the laptop and terraform state are gone. It then
# deletes itself. Its own role can do nothing but terminate instances.
resource "aws_iam_role" "scheduler_kill" {
  name_prefix = "loopgym-kill-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "scheduler.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "scheduler_kill" {
  name = "terminate-run-only"
  role = aws_iam_role.scheduler_kill.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ec2:TerminateInstances"
      Resource = "*"
    }]
  })
}

resource "aws_scheduler_schedule" "kill" {
  name_prefix = "loopgym-kill-${var.run_id}-"
  # One-time, at launch + max_hours + 15 min grace.
  schedule_expression = "at(${timeadd(timestamp(), "${var.max_hours}h15m")})"
  # No drift: the kill time is exact.
  flexible_time_window {
    mode = "OFF"
  }
  # Delete itself after firing (the run is over by then).
  action_after_completion = "DELETE"

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances"
    role_arn = aws_iam_role.scheduler_kill.arn
    input    = jsonencode({ instanceIds = aws_instance.worker[*].id })
  }
}
