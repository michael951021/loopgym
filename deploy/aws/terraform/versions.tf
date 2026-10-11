terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.68"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project = "loopgym"
      Run     = var.run_id
    }
  }
}
