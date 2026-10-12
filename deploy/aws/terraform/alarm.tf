# Idle kill: CPU < 5% for 6 x 5 min means the run finished or hung at launch;
# the automated action terminates the instance. CloudWatch's terminate action
# runs from AWS's own side (no laptop, no terraform state involved).
resource "aws_cloudwatch_metric_alarm" "idle" {
  count = var.nodes

  alarm_name          = "loopgym-idle-${var.run_id}-${count.index}"
  alarm_description   = "Terminate run ${var.run_id} worker ${count.index} when idle for 30 min."
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 6
  threshold           = 5
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  dimensions = {
    InstanceId = aws_instance.worker[count.index].id
  }
  alarm_actions = ["arn:aws:automate:${var.region}:ec2:terminate"]
}
