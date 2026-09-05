# Modules declare their provider requirement but never configure a provider:
# configuration is the root module's job, so the same module can be used
# against a different project or region without being edited.
terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}
