terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # GitLab-managed Terraform state, with locking. Everything is supplied at
  # init time via -backend-config=backend/<env>.hcl, so no project ID, state
  # name or credential is committed here.
  backend "http" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}
