# Budget backstop. Budgets data lags (up to ~24 h), so this is the LAST line,
# not the main cap: e-mails at 50/80% actual and 100% forecast, and at 100%
# actual the budget action attaches a deny-RunInstances policy to the launch
# role so no further instance can be started.
resource "aws_iam_policy" "deny_run_instances" {
  name_prefix = "loopgym-deny-run-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Deny"
      Action   = ["ec2:RunInstances", "ec2:RequestSpotInstances"]
      Resource = "*"
    }]
  })
}

# Role AWS Budgets assumes to attach the deny policy to the launch role when
# the budget action fires. It can do nothing else.
resource "aws_iam_role" "budget_action" {
  name_prefix = "loopgym-budget-action-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "budgets.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "budget_action" {
  name = "attach-deny-to-launch"
  role = aws_iam_role.budget_action.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "iam:PutRolePolicy"
      Resource = "arn:aws:iam::*:role/${var.launch_role_name}"
    }]
  })
}

resource "aws_budgets_budget" "run" {
  name         = "loopgym-${var.run_id}"
  time_unit    = "MONTHLY"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_usd)
  limit_unit   = "USD"

  notification {
    comparison_operator      = "GREATER_THAN"
    threshold                = 50
    threshold_type           = "ACTUAL"
    notification_type        = "EMAIL"
    subscriber_email_addresses = [var.budget_email]
  }
  notification {
    comparison_operator      = "GREATER_THAN"
    threshold                = 80
    threshold_type           = "ACTUAL"
    notification_type        = "EMAIL"
    subscriber_email_addresses = [var.budget_email]
  }
  notification {
    comparison_operator      = "GREATER_THAN"
    threshold                = 100
    threshold_type           = "FORECAST"
    notification_type        = "EMAIL"
    subscriber_email_addresses = [var.budget_email]
  }
}

resource "aws_budgets_budget_action" "deny" {
  budget_name        = aws_budgets_budget.run.name
  action_type        = "APPLY_IAM_POLICY"
  notification_type  = "ACTUAL"
  approval_model     = "AUTOMATIC"
  execution_role_arn = aws_iam_role.budget_action.arn

  action_threshold {
    action_threshold_type  = "PERCENTAGE"
    action_threshold_value = 100
  }

  definition {
    iam_action_definition {
      policy_arn = aws_iam_policy.deny_run_instances.arn
      roles      = [var.launch_role_name]
    }
  }

  subscriber {
    subscription_type = "EMAIL"
    address           = var.budget_email
  }
}
