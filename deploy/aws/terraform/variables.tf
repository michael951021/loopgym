variable "run_id" {
  type        = string
  description = "Run label; must be [a-zA-Z0-9-]."
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "nodes" {
  type    = number
  default = 2

  validation {
    condition     = var.nodes == 1 || var.nodes == 2
    error_message = "nodes must be 1 or 2 (study is 2 x g4dn.xlarge; quota is 8 vCPU)."
  }
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"

  validation {
    condition     = var.instance_type == "g4dn.xlarge"
    error_message = "only g4dn.xlarge is approved for the scaling study."
  }
}

variable "max_price" {
  type        = number
  default     = 0.30
  description = "Spot max price per hour (USD). g4dn.xlarge on-demand is ~$0.52."

  validation {
    condition     = var.max_price > 0 && var.max_price <= 0.60
    error_message = "max_price must be in (0, 0.60]."
  }
}

variable "max_hours" {
  type        = number
  default     = 3
  description = "Hard wall-clock cap on the run, in whole hours."

  validation {
    condition     = var.max_hours >= 1 && var.max_hours <= 3 && var.max_hours == floor(var.max_hours)
    error_message = "max_hours must be a whole number in 1..3."
  }
}

variable "budget_usd" {
  type    = number
  default = 10

  validation {
    condition     = var.budget_usd > 0 && var.budget_usd <= 10
    error_message = "budget_usd must be in (0, 10]; the account budget is $10 ever."
  }
}

variable "budget_email" {
  type = string
}

variable "launch_role_name" {
  type        = string
  default     = "loopgym-launch"
  description = "IAM role the budget action attaches the deny policy to."
}
