---
title: "Terraform & Cloud Infrastructure"
---

# Chapter 6B: Terraform & Cloud Infrastructure

While Kubernetes is excellent for orchestrating containers, we still need to provision the underlying infrastructure - servers, networks, databases, and load balancers. This is where Terraform comes in. In this chapter, I'll show you how to deploy ResearcherAI's complete infrastructure on DigitalOcean using Infrastructure as Code.

## Why Infrastructure as Code?

Before Terraform, deploying infrastructure meant:
- Clicking through cloud provider dashboards
- Copy-pasting configurations
- Inconsistent environments (dev vs prod)
- No version control
- Manual documentation
- "Works on my machine" problems

With Terraform:
- Infrastructure defined in code files
- Version controlled with Git
- Reproducible deployments
- Automated provisioning
- Infrastructure reviews (pull requests)
- Disaster recovery (redeploy from code)

### Web Developer Analogy

**Manual Infrastructure** = Directly editing production database
- Click buttons in admin panel
- Hope you remember what you changed
- No audit trail

**Infrastructure as Code** = Database migrations
- Changes defined in code
- Version controlled
- Reviewable
- Rollback-able
- Documented automatically

## Terraform Basics

Terraform is like npm for infrastructure:

```hcl
# Terraform configuration
resource "digitalocean_droplet" "app_server" {
  name   = "app-server-1"
  size   = "s-2vcpu-4gb"
  image  = "ubuntu-22-04-x64"
  region = "nyc3"
}
```

```javascript
// JavaScript equivalent (conceptual)
const appServer = new DigitalOceanDroplet({
  name: 'app-server-1',
  size: 's-2vcpu-4gb',
  image: 'ubuntu-22-04-x64',
  region: 'nyc3'
});
```

### Core Concepts

**1. Providers** (Like npm packages)
```hcl
terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}
```

**2. Resources** (Infrastructure components)
```hcl
resource "type" "name" {
  argument1 = "value1"
  argument2 = "value2"
}
```

**3. Variables** (Like function parameters)
```hcl
variable "droplet_count" {
  type    = number
  default = 2
}
```

**4. Outputs** (Return values)
```hcl
output "server_ip" {
  value = digitalocean_droplet.app_server.ipv4_address
}
```

**5. Modules** (Reusable components)
```hcl
module "vpc" {
  source = "./modules/vpc"
  name   = "production-vpc"
}
```

## ResearcherAI Infrastructure Architecture

Here's what we're building with Terraform:

```
Internet
  ↓
Load Balancer (DigitalOcean LB)
  ↓
┌─────────────── VPC (10.10.0.0/16) ───────────────┐
│                                                    │
│  ┌──────────────┐      ┌──────────────┐         │
│  │ App Server 1 │      │ App Server 2 │         │
│  │ Docker       │      │ Docker       │         │
│  │ Compose      │      │ Compose      │         │
│  └──────────────┘      └──────────────┘         │
│         │                     │                   │
│         └─────────┬───────────┘                  │
│                   ↓                               │
│  ┌─────────────────────────────────┐            │
│  │   Managed PostgreSQL Cluster    │            │
│  │   (Metadata & Sessions)          │            │
│  └─────────────────────────────────┘            │
│                                                    │
│  ┌─────────────────────────────────┐            │
│  │   Managed Kafka Cluster          │            │
│  │   (3 nodes for HA)               │            │
│  └─────────────────────────────────┘            │
│                                                    │
│  ┌──────────────┐      ┌──────────────┐         │
│  │ Neo4j        │      │ Qdrant       │         │
│  │ (on Docker)  │      │ (on Docker)  │         │
│  └──────────────┘      └──────────────┘         │
│                                                    │
└────────────────────────────────────────────────────┘

Firewall Rules:
- Internet → Load Balancer: 80, 443
- Load Balancer → App Servers: 8000
- App Servers → Databases: Internal only
- App Servers → Internet: API calls
```

## Project Structure

```
terraform/
├── main.tf                    # Main infrastructure
├── variables.tf               # Input variables
├── outputs.tf                 # Output values
├── terraform.tfvars.example   # Configuration template
├── modules/
│   ├── vpc/                   # Virtual Private Cloud
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── droplet/               # Application servers
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── database/              # PostgreSQL
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── kafka/                 # Kafka cluster
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── loadbalancer/          # Load balancer
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── firewall/              # Security rules
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── environments/
│   ├── dev/
│   │   └── terraform.tfvars
│   ├── staging/
│   │   └── terraform.tfvars
│   └── production/
│       └── terraform.tfvars
└── scripts/
    └── init_app_server.sh     # Server setup script
```

## Building the Infrastructure

### Step 1: Main Configuration

`main.tf` - The entry point:

```hcl
terraform {
  required_version = ">= 1.0"

  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }

  # Optional: Remote state in S3/Spaces
  # backend "s3" {
  #   endpoint                    = "nyc3.digitaloceanspaces.com"
  #   key                         = "terraform.tfstate"
  #   bucket                      = "researcherai-terraform"
  #   region                      = "us-east-1"
  #   skip_credentials_validation = true
  #   skip_metadata_api_check     = true
  # }
}

provider "digitalocean" {
  token = var.do_token
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  project_name = var.project_name
  environment  = var.environment
  region       = var.region
  vpc_cidr     = var.vpc_cidr
}

# Droplet Module (Application Servers)
module "droplet" {
  source = "./modules/droplet"

  project_name    = var.project_name
  environment     = var.environment
  region          = var.region
  droplet_count   = var.droplet_count
  droplet_size    = var.droplet_size
  ssh_key_ids     = var.ssh_key_ids
  vpc_uuid        = module.vpc.vpc_id

  # Environment variables for app
  google_api_key  = var.google_api_key
  neo4j_uri       = var.external_neo4j_uri != "" ? var.external_neo4j_uri : "bolt://localhost:7687"
  neo4j_password  = var.neo4j_password
  qdrant_url      = var.external_qdrant_url != "" ? var.external_qdrant_url : "http://localhost:6333"
  use_kafka       = var.use_managed_kafka
  kafka_bootstrap = var.use_managed_kafka ? module.kafka[0].bootstrap_servers : "localhost:9092"

  depends_on = [module.vpc]
}

# Database Module (PostgreSQL)
module "database" {
  source = "./modules/database"

  project_name = var.project_name
  environment  = var.environment
  region       = var.region
  db_size      = var.db_size
  db_node_count = var.db_node_count
  vpc_uuid     = module.vpc.vpc_id

  depends_on = [module.vpc]
}

# Kafka Module (if using managed Kafka)
module "kafka" {
  count  = var.use_managed_kafka ? 1 : 0
  source = "./modules/kafka"

  project_name = var.project_name
  environment  = var.environment
  region       = var.region
  kafka_size   = var.kafka_size
  kafka_nodes  = var.kafka_node_count
  vpc_uuid     = module.vpc.vpc_id

  depends_on = [module.vpc]
}

# Load Balancer Module
module "loadbalancer" {
  source = "./modules/loadbalancer"

  project_name = var.project_name
  environment  = var.environment
  region       = var.region
  vpc_uuid     = module.vpc.vpc_id
  droplet_ids  = module.droplet.droplet_ids

  depends_on = [module.droplet]
}

# Firewall Module
module "firewall" {
  source = "./modules/firewall"

  project_name      = var.project_name
  environment       = var.environment
  droplet_ids       = module.droplet.droplet_ids
  loadbalancer_id   = module.loadbalancer.lb_id
  vpc_cidr          = var.vpc_cidr
  allowed_ssh_ips   = var.allowed_ssh_ips

  depends_on = [module.droplet, module.loadbalancer]
}

# DigitalOcean Project (for organization)
resource "digitalocean_project" "researcherai" {
  name        = "${var.project_name}-${var.environment}"
  description = "ResearcherAI Multi-Agent RAG System - ${var.environment}"
  purpose     = "Web Application"
  environment = var.environment

  resources = concat(
    module.droplet.droplet_urns,
    [module.loadbalancer.lb_urn],
    [module.database.db_urn],
    var.use_managed_kafka ? [module.kafka[0].kafka_urn] : []
  )
}
```

### Step 2: Variables

`variables.tf` - All configurable parameters:

```hcl
# DigitalOcean API Token
variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

# Project Configuration
variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "researcherai"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "nyc3"

  validation {
    condition = contains([
      "nyc1", "nyc3", "sfo1", "sfo2", "sfo3",
      "ams3", "sgp1", "lon1", "fra1", "tor1", "blr1"
    ], var.region)
    error_message = "Invalid DigitalOcean region."
  }
}

# Network Configuration
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.10.0.0/16"
}

# Droplet Configuration
variable "droplet_count" {
  description = "Number of application servers"
  type        = number
  default     = 2

  validation {
    condition     = var.droplet_count >= 1 && var.droplet_count <= 10
    error_message = "Droplet count must be between 1 and 10."
  }
}

variable "droplet_size" {
  description = "Droplet size slug"
  type        = string
  default     = "s-2vcpu-4gb"  # $36/month per server

  validation {
    condition = contains([
      "s-1vcpu-2gb",   # $18/mo - Dev/staging
      "s-2vcpu-4gb",   # $36/mo - Production
      "s-4vcpu-8gb",   # $72/mo - High traffic
      "s-8vcpu-16gb"   # $144/mo - Very high traffic
    ], var.droplet_size)
    error_message = "Invalid droplet size."
  }
}

variable "ssh_key_ids" {
  description = "SSH key IDs for droplet access"
  type        = list(string)
}

# Database Configuration
variable "db_size" {
  description = "Database cluster size"
  type        = string
  default     = "db-s-2vcpu-4gb"  # $60/month
}

variable "db_node_count" {
  description = "Number of database nodes (1 or 2)"
  type        = number
  default     = 1

  validation {
    condition     = var.db_node_count >= 1 && var.db_node_count <= 2
    error_message = "Database node count must be 1 or 2."
  }
}

# Kafka Configuration
variable "use_managed_kafka" {
  description = "Use DigitalOcean managed Kafka"
  type        = bool
  default     = true
}

variable "kafka_size" {
  description = "Kafka node size"
  type        = string
  default     = "db-s-2vcpu-2gb"  # $30/month per node
}

variable "kafka_node_count" {
  description = "Number of Kafka nodes (minimum 3 for HA)"
  type        = number
  default     = 3

  validation {
    condition     = var.kafka_node_count >= 3
    error_message = "Kafka requires minimum 3 nodes for high availability."
  }
}

# Application Secrets
variable "google_api_key" {
  description = "Google Gemini API key"
  type        = string
  sensitive   = true
}

variable "neo4j_password" {
  description = "Neo4j database password"
  type        = string
  sensitive   = true
  default     = "secure-neo4j-password"
}

# External Services (optional)
variable "external_neo4j_uri" {
  description = "External Neo4j URI (if not using local)"
  type        = string
  default     = ""
}

variable "external_qdrant_url" {
  description = "External Qdrant URL (if not using local)"
  type        = string
  default     = ""
}

# Security
variable "allowed_ssh_ips" {
  description = "IP addresses allowed SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # WARNING: Restrict in production!
}

# Feature Flags
variable "enable_monitoring" {
  description = "Enable DigitalOcean monitoring"
  type        = bool
  default     = true
}

variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

# Tags
variable "custom_tags" {
  description = "Custom tags for resources"
  type        = list(string)
  default     = []
}
```

### Step 3: VPC Module

`modules/vpc/main.tf` - Private network:

```hcl
resource "digitalocean_vpc" "main" {
  name     = "${var.project_name}-vpc-${var.environment}"
  region   = var.region
  ip_range = var.vpc_cidr

  description = "VPC for ${var.project_name} ${var.environment} environment"
}

# Outputs
output "vpc_id" {
  value = digitalocean_vpc.main.id
}

output "vpc_urn" {
  value = digitalocean_vpc.main.urn
}
```

`modules/vpc/variables.tf`:

```hcl
variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "region" {
  type = string
}

variable "vpc_cidr" {
  type    = string
  default = "10.10.0.0/16"
}
```

### Step 4: Droplet Module

`modules/droplet/main.tf` - Application servers:

```hcl
resource "digitalocean_droplet" "app_server" {
  count  = var.droplet_count
  name   = "${var.project_name}-app-${var.environment}-${count.index + 1}"
  region = var.region
  size   = var.droplet_size
  image  = "ubuntu-22-04-x64"

  # SSH keys
  ssh_keys = var.ssh_key_ids

  # VPC
  vpc_uuid = var.vpc_uuid

  # IPv6
  ipv6 = true

  # Monitoring
  monitoring = true

  # Backups
  backups = var.enable_backups

  # User data - initialization script
  user_data = templatefile("${path.module}/../../scripts/init_app_server.sh", {
    google_api_key  = var.google_api_key
    neo4j_uri       = var.neo4j_uri
    neo4j_password  = var.neo4j_password
    qdrant_url      = var.qdrant_url
    kafka_bootstrap = var.kafka_bootstrap
    use_neo4j       = "true"
    use_qdrant      = "true"
    use_kafka       = var.use_kafka ? "true" : "false"
  })

  # Optional volumes for persistent data
  dynamic "volume_ids" {
    for_each = var.attach_volumes ? [1] : []
    content {
      volume_ids = [digitalocean_volume.data[count.index].id]
    }
  }

  # Tags
  tags = concat(
    [var.project_name, var.environment, "app-server"],
    var.custom_tags
  )
}

# Optional persistent volumes
resource "digitalocean_volume" "data" {
  count                   = var.attach_volumes ? var.droplet_count : 0
  region                  = var.region
  name                    = "${var.project_name}-data-${var.environment}-${count.index + 1}"
  size                    = var.volume_size
  initial_filesystem_type = "ext4"
  description             = "Data volume for ${var.project_name} app server ${count.index + 1}"

  tags = [var.project_name, var.environment, "data-volume"]
}

# Outputs
output "droplet_ids" {
  value = digitalocean_droplet.app_server[*].id
}

output "droplet_urns" {
  value = digitalocean_droplet.app_server[*].urn
}

output "droplet_ips" {
  value = digitalocean_droplet.app_server[*].ipv4_address
}

output "droplet_private_ips" {
  value = digitalocean_droplet.app_server[*].ipv4_address_private
}
```

### Step 5: Server Initialization Script

`scripts/init_app_server.sh` - Automated setup:

```bash
#!/bin/bash
set -e

echo "=== ResearcherAI Server Initialization ==="

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl enable docker
systemctl start docker

# Install Docker Compose
echo "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install dependencies
apt-get install -y git python3-pip nginx

# Create application directory
mkdir -p /opt/researcherai
cd /opt/researcherai

# Create environment file
cat > /opt/researcherai/.env <<'EOF'
# API Keys
GOOGLE_API_KEY=${google_api_key}

# Neo4j Configuration
USE_NEO4J=${use_neo4j}
NEO4J_URI=${neo4j_uri}
NEO4J_USER=neo4j
NEO4J_PASSWORD=${neo4j_password}

# Qdrant Configuration
USE_QDRANT=${use_qdrant}
QDRANT_HOST=${qdrant_url}

# Kafka Configuration
USE_KAFKA=${use_kafka}
KAFKA_BOOTSTRAP_SERVERS=${kafka_bootstrap}

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF

# Setup docker-compose as systemd service
cat > /etc/systemd/system/researcherai.service <<'EOF'
[Unit]
Description=ResearcherAI Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/researcherai
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Setup Nginx reverse proxy
cat > /etc/nginx/sites-available/researcherai <<'EOF'
server {
    listen 8000;
    server_name _;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://localhost:8001/health;
        access_log off;
    }
}
EOF

ln -sf /etc/nginx/sites-available/researcherai /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# Enable and start service
systemctl daemon-reload
systemctl enable researcherai.service

echo "=== Server initialization complete ==="
echo "Application directory: /opt/researcherai"
echo "Next steps:"
echo "1. Clone your repository to /opt/researcherai"
echo "2. Start services: systemctl start researcherai"
```

### Step 6: Database Module

`modules/database/main.tf` - Managed PostgreSQL:

```hcl
resource "digitalocean_database_cluster" "postgres" {
  name       = "${var.project_name}-db-${var.environment}"
  engine     = "pg"
  version    = "15"
  size       = var.db_size
  region     = var.region
  node_count = var.db_node_count

  # Private network only
  private_network_uuid = var.vpc_uuid

  maintenance_window {
    day  = "sunday"
    hour = "02:00:00"
  }

  tags = [var.project_name, var.environment, "database"]
}

# Default database
resource "digitalocean_database_db" "app_metadata" {
  cluster_id = digitalocean_database_cluster.postgres.id
  name       = "app_metadata"
}

# Default user
resource "digitalocean_database_user" "app_user" {
  cluster_id = digitalocean_database_cluster.postgres.id
  name       = "app_user"
}

# Firewall - allow from VPC only
resource "digitalocean_database_firewall" "postgres_fw" {
  cluster_id = digitalocean_database_cluster.postgres.id

  rule {
    type  = "tag"
    value = var.project_name
  }
}

# Outputs
output "db_urn" {
  value = digitalocean_database_cluster.postgres.urn
}

output "db_host" {
  value     = digitalocean_database_cluster.postgres.private_host
  sensitive = true
}

output "db_port" {
  value = digitalocean_database_cluster.postgres.port
}

output "db_name" {
  value = digitalocean_database_db.app_metadata.name
}

output "db_user" {
  value     = digitalocean_database_user.app_user.name
  sensitive = true
}

output "db_password" {
  value     = digitalocean_database_user.app_user.password
  sensitive = true
}

output "connection_string" {
  value     = digitalocean_database_cluster.postgres.private_uri
  sensitive = true
}
```

### Step 7: Kafka Module

`modules/kafka/main.tf` - Managed Kafka cluster:

```hcl
resource "digitalocean_database_cluster" "kafka" {
  name       = "${var.project_name}-kafka-${var.environment}"
  engine     = "kafka"
  version    = "3.5"
  size       = var.kafka_size
  region     = var.region
  node_count = var.kafka_nodes  # Minimum 3 for HA

  # Private network only
  private_network_uuid = var.vpc_uuid

  maintenance_window {
    day  = "sunday"
    hour = "03:00:00"
  }

  tags = [var.project_name, var.environment, "kafka"]
}

# Default Kafka user
resource "digitalocean_database_user" "kafka_app_user" {
  cluster_id = digitalocean_database_cluster.kafka.id
  name       = "kafka_app_user"
}

# Kafka topics (create via API or application)
resource "digitalocean_database_kafka_topic" "events" {
  cluster_id         = digitalocean_database_cluster.kafka.id
  name               = "query.submitted"
  partition_count    = 3
  replication_factor = 3

  config {
    cleanup_policy                      = "delete"
    compression_type                    = "producer"
    delete_retention_ms                 = 86400000  # 1 day
    file_delete_delay_ms                = 60000
    flush_messages                      = 9223372036854775807
    flush_ms                            = 9223372036854775807
    index_interval_bytes                = 4096
    max_compaction_lag_ms               = 9223372036854775807
    max_message_bytes                   = 1048588
    message_down_conversion_enable      = true
    message_format_version              = "3.0-IV1"
    message_timestamp_difference_max_ms = 9223372036854775807
    message_timestamp_type              = "create_time"
    min_cleanable_dirty_ratio           = 0.5
    min_compaction_lag_ms               = 0
    min_insync_replicas                 = 2
    preallocate                         = false
    retention_bytes                     = -1
    retention_ms                        = 604800000  # 7 days
    segment_bytes                       = 1073741824
    segment_index_bytes                 = 10485760
    segment_jitter_ms                   = 0
    segment_ms                          = 604800000
  }
}

# Firewall - allow from VPC only
resource "digitalocean_database_firewall" "kafka_fw" {
  cluster_id = digitalocean_database_cluster.kafka.id

  rule {
    type  = "tag"
    value = var.project_name
  }
}

# Outputs
output "kafka_urn" {
  value = digitalocean_database_cluster.kafka.urn
}

output "bootstrap_servers" {
  value     = digitalocean_database_cluster.kafka.private_host
  sensitive = true
}

output "kafka_port" {
  value = digitalocean_database_cluster.kafka.port
}

output "kafka_username" {
  value     = digitalocean_database_user.kafka_app_user.name
  sensitive = true
}

output "kafka_password" {
  value     = digitalocean_database_user.kafka_app_user.password
  sensitive = true
}
```

### Step 8: Load Balancer Module

`modules/loadbalancer/main.tf` - Traffic distribution:

```hcl
resource "digitalocean_loadbalancer" "main" {
  name   = "${var.project_name}-lb-${var.environment}"
  region = var.region

  # VPC
  vpc_uuid = var.vpc_uuid

  # HTTP forwarding
  forwarding_rule {
    entry_protocol  = "http"
    entry_port      = 80
    target_protocol = "http"
    target_port     = 8000
  }

  # HTTPS forwarding (if you have SSL certificate)
  forwarding_rule {
    entry_protocol  = "https"
    entry_port      = 443
    target_protocol = "http"
    target_port     = 8000

    # Optional: Add your SSL certificate ID
    # certificate_id = var.ssl_certificate_id
  }

  # Health check
  healthcheck {
    protocol               = "http"
    port                   = 8000
    path                   = "/health"
    check_interval_seconds = 10
    response_timeout_seconds = 5
    unhealthy_threshold    = 3
    healthy_threshold      = 3
  }

  # Sticky sessions
  sticky_sessions {
    type               = "cookies"
    cookie_name        = "lb-session"
    cookie_ttl_seconds = 3600
  }

  # Droplet IDs
  droplet_ids = var.droplet_ids

  # Tags
  droplet_tag = "${var.project_name}-${var.environment}"

  # Algorithm
  algorithm = "round_robin"
}

# Outputs
output "lb_id" {
  value = digitalocean_loadbalancer.main.id
}

output "lb_urn" {
  value = digitalocean_loadbalancer.main.urn
}

output "lb_ip" {
  value = digitalocean_loadbalancer.main.ip
}

output "lb_url" {
  value = "http://${digitalocean_loadbalancer.main.ip}"
}
```

### Step 9: Firewall Module

`modules/firewall/main.tf` - Security rules:

```hcl
resource "digitalocean_firewall" "app_firewall" {
  name = "${var.project_name}-firewall-${var.environment}"

  # Apply to all application servers
  droplet_ids = var.droplet_ids

  # Inbound rules

  # Allow HTTP/HTTPS from load balancer
  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_load_balancer_uids = [var.loadbalancer_id]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_load_balancer_uids = [var.loadbalancer_id]
  }

  # Allow app port from load balancer
  inbound_rule {
    protocol         = "tcp"
    port_range       = "8000"
    source_load_balancer_uids = [var.loadbalancer_id]
  }

  # Allow SSH from specific IPs
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.allowed_ssh_ips
  }

  # Allow all from VPC (internal communication)
  inbound_rule {
    protocol         = "tcp"
    port_range       = "1-65535"
    source_addresses = [var.vpc_cidr]
  }

  inbound_rule {
    protocol         = "udp"
    port_range       = "1-65535"
    source_addresses = [var.vpc_cidr]
  }

  # Outbound rules

  # Allow all TCP (for internet access)
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  # Allow all UDP
  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  # Allow ICMP
  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
```

### Step 10: Outputs

`outputs.tf` - Information after deployment:

```hcl
# VPC Information
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

# Server Information
output "app_server_ips" {
  description = "Application server public IPs"
  value       = module.droplet.droplet_ips
}

output "app_server_private_ips" {
  description = "Application server private IPs"
  value       = module.droplet.droplet_private_ips
}

# Load Balancer
output "load_balancer_ip" {
  description = "Load balancer public IP"
  value       = module.loadbalancer.lb_ip
}

output "application_url" {
  description = "Application URL"
  value       = module.loadbalancer.lb_url
}

# Database (marked sensitive)
output "database_host" {
  description = "Database host"
  value       = module.database.db_host
  sensitive   = true
}

output "database_connection_string" {
  description = "Database connection string"
  value       = module.database.connection_string
  sensitive   = true
}

# Kafka (if enabled)
output "kafka_bootstrap_servers" {
  description = "Kafka bootstrap servers"
  value       = var.use_managed_kafka ? module.kafka[0].bootstrap_servers : "Using Docker Kafka"
  sensitive   = true
}

# SSH Commands
output "ssh_commands" {
  description = "SSH commands for each server"
  value = [
    for ip in module.droplet.droplet_ips :
    "ssh root@${ip}"
  ]
}

# Deployment Information
output "deployment_info" {
  description = "Deployment summary"
  value = {
    environment    = var.environment
    region         = var.region
    server_count   = var.droplet_count
    server_size    = var.droplet_size
    database       = "PostgreSQL ${var.db_size}"
    kafka_enabled  = var.use_managed_kafka
    load_balancer  = module.loadbalancer.lb_ip
  }
}

# Estimated Monthly Cost
output "estimated_monthly_cost" {
  description = "Estimated monthly cost in USD"
  value = format("$%.2f",
    var.droplet_count * (
      var.droplet_size == "s-1vcpu-2gb" ? 18 :
      var.droplet_size == "s-2vcpu-4gb" ? 36 :
      var.droplet_size == "s-4vcpu-8gb" ? 72 : 144
    ) +
    (var.db_size == "db-s-1vcpu-1gb" ? 15 :
     var.db_size == "db-s-2vcpu-4gb" ? 60 : 120) +
    (var.use_managed_kafka ? var.kafka_node_count * 30 : 0) +
    10  # Load balancer
  )
}

# Next Steps
output "next_steps" {
  description = "Next steps after deployment"
  value = <<-EOT

    Deployment Complete! 🎉

    1. Access your servers:
       ${join("\n       ", [for ip in module.droplet.droplet_ips : "ssh root@${ip}"])}

    2. Clone your repository:
       cd /opt/researcherai
       git clone <your-repo-url> .

    3. Start the application:
       systemctl start researcherai

    4. Check logs:
       journalctl -u researcherai -f

    5. Access the application:
       ${module.loadbalancer.lb_url}

    6. Monitor resources:
       doctl compute droplet list
       doctl database list

    Estimated monthly cost: ${format("$%.2f",
      var.droplet_count * (
        var.droplet_size == "s-1vcpu-2gb" ? 18 :
        var.droplet_size == "s-2vcpu-4gb" ? 36 :
        var.droplet_size == "s-4vcpu-8gb" ? 72 : 144
      ) +
      (var.db_size == "db-s-1vcpu-1gb" ? 15 :
       var.db_size == "db-s-2vcpu-4gb" ? 60 : 120) +
      (var.use_managed_kafka ? var.kafka_node_count * 30 : 0) +
      10
    )}
  EOT
}
```

## Deploying with Terraform

### Step 1: Prerequisites

```bash
# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
terraform version

# Install DigitalOcean CLI (optional)
curl -sL https://github.com/digitalocean/doctl/releases/download/v1.98.1/doctl-1.98.1-linux-amd64.tar.gz | tar -xzv
sudo mv doctl /usr/local/bin/

# Authenticate doctl
doctl auth init
```

### Step 2: Get DigitalOcean API Token

```bash
# Create API token at:
# https://cloud.digitalocean.com/account/api/tokens

# Save to environment variable
export DO_TOKEN="your-digitalocean-api-token-here"
```

### Step 3: Add SSH Key

```bash
# Generate SSH key if you don't have one
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# Add to DigitalOcean
doctl compute ssh-key import my-key --public-key-file ~/.ssh/id_rsa.pub

# Get SSH key ID
doctl compute ssh-key list
```

### Step 4: Create Configuration

```bash
cd terraform/

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
nano terraform.tfvars
```

`terraform.tfvars`:

```hcl
# DigitalOcean Configuration
do_token = "dop_v1_your_token_here"

# Project Settings
project_name = "researcherai"
environment  = "production"
region       = "nyc3"

# Server Configuration
droplet_count = 2
droplet_size  = "s-2vcpu-4gb"  # $36/month per server
ssh_key_ids   = ["12345678"]    # Your SSH key ID

# Database Configuration
db_size       = "db-s-2vcpu-4gb"  # $60/month
db_node_count = 1

# Kafka Configuration
use_managed_kafka = true
kafka_size        = "db-s-2vcpu-2gb"  # $30/month per node
kafka_node_count  = 3                 # $90/month total

# Application Secrets
google_api_key = "AIzaSy..."
neo4j_password = "secure-password-here"

# Security
allowed_ssh_ips = ["your.ip.address.here/32"]  # Your IP only

# Features
enable_monitoring = true
enable_backups    = true
```

### Step 5: Initialize Terraform

```bash
# Initialize (downloads providers)
terraform init

# Output:
# Initializing modules...
# Initializing the backend...
# Initializing provider plugins...
# - Finding digitalocean/digitalocean versions matching "~> 2.0"...
# - Installing digitalocean/digitalocean v2.32.0...
# Terraform has been successfully initialized!
```

### Step 6: Plan Deployment

```bash
# See what will be created
terraform plan

# Output shows all resources:
# Terraform will perform the following actions:
#
#   # module.droplet.digitalocean_droplet.app_server[0] will be created
#   + resource "digitalocean_droplet" "app_server" {
#       + name   = "researcherai-app-production-1"
#       + region = "nyc3"
#       + size   = "s-2vcpu-4gb"
#       ...
#   }
#
# Plan: 15 to add, 0 to change, 0 to destroy.
```

### Step 7: Apply Configuration

```bash
# Deploy infrastructure
terraform apply

# Review plan and type 'yes' to confirm

# Deployment takes 5-10 minutes
# Watch progress in terminal
```

### Step 8: Get Outputs

```bash
# View all outputs
terraform output

# Example output:
# app_server_ips = [
#   "167.99.123.45",
#   "167.99.123.46",
# ]
# application_url = "http://174.138.45.67"
# estimated_monthly_cost = "$232.00"
# load_balancer_ip = "174.138.45.67"

# Get specific output
terraform output application_url

# Get sensitive output
terraform output -json database_connection_string
```

### Step 9: SSH to Servers

```bash
# Get SSH commands
terraform output -json ssh_commands

# SSH to first server
ssh root@$(terraform output -json app_server_ips | jq -r '.[0]')

# Check application status
cd /opt/researcherai
systemctl status researcherai
docker-compose ps
```

### Step 10: Deploy Application

```bash
# On each server:
cd /opt/researcherai

# Clone your repository
git clone https://github.com/your-username/ResearcherAI.git .

# Start services
systemctl start researcherai

# Check logs
journalctl -u researcherai -f

# Or with docker-compose directly
docker-compose logs -f
```

## Multi-Environment Setup

### Directory Structure

```
terraform/
├── environments/
│   ├── dev/
│   │   └── terraform.tfvars
│   ├── staging/
│   │   └── terraform.tfvars
│   └── production/
│       └── terraform.tfvars
```

### Development Environment

`environments/dev/terraform.tfvars`:

```hcl
project_name = "researcherai"
environment  = "dev"
region       = "nyc3"

# Minimal resources for dev
droplet_count = 1
droplet_size  = "s-1vcpu-2gb"  # $18/month

db_size       = "db-s-1vcpu-1gb"  # $15/month
db_node_count = 1

# Use Docker Kafka instead of managed
use_managed_kafka = false

# Dev secrets
google_api_key = "AIzaSy...dev-key"
neo4j_password = "dev-password"

# Allow SSH from anywhere in dev
allowed_ssh_ips = ["0.0.0.0/0"]

# Disable backups in dev
enable_backups = false
```

### Deploy Development

```bash
cd terraform/

# Use dev configuration
terraform workspace new dev
terraform workspace select dev

# Plan with dev vars
terraform plan -var-file=environments/dev/terraform.tfvars

# Apply
terraform apply -var-file=environments/dev/terraform.tfvars
```

### Production Environment

`environments/production/terraform.tfvars`:

```hcl
project_name = "researcherai"
environment  = "production"
region       = "nyc3"

# Production resources
droplet_count = 4
droplet_size  = "s-4vcpu-8gb"  # $72/month × 4 = $288/month

db_size       = "db-s-4vcpu-8gb"  # $120/month
db_node_count = 2  # Standby for HA

# Managed Kafka with HA
use_managed_kafka = true
kafka_size        = "db-s-2vcpu-2gb"
kafka_node_count  = 3  # $90/month

# Production secrets (use environment variables!)
google_api_key = "${GOOGLE_API_KEY}"
neo4j_password = "${NEO4J_PASSWORD}"

# Restrict SSH to office/VPN
allowed_ssh_ips = ["203.0.113.0/24"]

# Enable all production features
enable_monitoring = true
enable_backups    = true
```

### Deploy Production

```bash
# Use production workspace
terraform workspace new production
terraform workspace select production

# Store secrets in environment
export TF_VAR_google_api_key="your-prod-key"
export TF_VAR_neo4j_password="secure-prod-password"

# Plan
terraform plan -var-file=environments/production/terraform.tfvars

# Apply with extra caution
terraform apply -var-file=environments/production/terraform.tfvars
```

## Managing Infrastructure

### Update Infrastructure

```bash
# Make changes to .tf files or terraform.tfvars

# Preview changes
terraform plan

# Apply changes
terraform apply

# Terraform will:
# - Create new resources (green +)
# - Update existing resources (yellow ~)
# - Destroy/recreate resources (red -/+)
```

### Scale Up/Down

```bash
# Edit terraform.tfvars
droplet_count = 4  # Increase from 2 to 4

# Apply changes
terraform apply

# Terraform will create 2 new droplets
# Load balancer automatically includes them
```

### Upgrade Server Size

```bash
# Edit terraform.tfvars
droplet_size = "s-4vcpu-8gb"  # Upgrade from s-2vcpu-4gb

# Apply (WARNING: Recreates droplets!)
terraform apply

# For zero-downtime:
# 1. Increase droplet_count first
# 2. Change droplet_size
# 3. Wait for new servers
# 4. Decrease droplet_count
```

### View State

```bash
# List all resources
terraform state list

# Show specific resource
terraform state show module.droplet.digitalocean_droplet.app_server[0]

# Show all outputs
terraform output
```

### Destroy Infrastructure

```bash
# Preview destruction
terraform plan -destroy

# Destroy everything (DANGEROUS!)
terraform destroy

# Destroy specific resource
terraform destroy -target=module.droplet.digitalocean_droplet.app_server[0]
```

## Cost Management

### Cost Breakdown

**Default Production Configuration:**

```
2× Application Servers (s-2vcpu-4gb)     $72/month
1× PostgreSQL Database (db-s-2vcpu-4gb)  $60/month
3× Kafka Nodes (db-s-2vcpu-2gb)          $90/month
1× Load Balancer                         $10/month
────────────────────────────────────────────────────
Total                                    $232/month
```

**Budget Options:**

```
Development ($43/month):
- 1× s-1vcpu-2gb droplet       $18
- 1× db-s-1vcpu-1gb database   $15
- Kafka on Docker              $0
- Load balancer                $10

Production ($232/month):
- 2× s-2vcpu-4gb droplets      $72
- 1× db-s-2vcpu-4gb database   $60
- 3× Kafka nodes               $90
- Load balancer                $10

High-Traffic ($628/month):
- 4× s-4vcpu-8gb droplets      $288
- 2× db-s-4vcpu-8gb database   $240
- 3× Kafka nodes               $90
- Load balancer                $10
```

### Cost Optimization Tips

1. **Use development configs for dev/staging**
2. **Enable auto-shutdown for non-production** (manual via API)
3. **Use reserved instances** (DigitalOcean doesn't offer, but AWS/GCP do)
4. **Monitor with alerts** to catch runaway costs
5. **Right-size resources** based on actual usage

### Monitoring Costs

```bash
# DigitalOcean CLI
doctl balance get

# Monthly usage
doctl invoice list

# Resource costs
doctl compute droplet list --format Name,Size,Price
doctl database list --format Name,Size,Price
```

## Backup and Disaster Recovery

### Automated Backups

```hcl
# Enable in terraform.tfvars
enable_backups = true

# Droplets: Weekly backups, 4 weeks retention
# Databases: Daily backups, 7 days retention
```

### Manual Snapshots

```bash
# Create droplet snapshot
doctl compute droplet-action snapshot <droplet-id> --snapshot-name "before-upgrade"

# Create database backup
doctl database backup list <database-id>
doctl database backup create <database-id>
```

### Disaster Recovery Plan

```bash
# 1. Store Terraform state safely (S3/Spaces)
terraform {
  backend "s3" {
    endpoint = "nyc3.digitaloceanspaces.com"
    bucket   = "researcherai-terraform-state"
    key      = "production/terraform.tfstate"
    region   = "us-east-1"
    skip_credentials_validation = true
  }
}

# 2. Backup secrets separately (password manager)

# 3. Document recovery procedure
# If complete disaster:
# - Clone Git repository
# - Configure terraform.tfvars with secrets
# - Run: terraform init -backend-config=backend.tfvars
# - Run: terraform apply
# - Restore database from backup
# - Deploy application code
```

## Troubleshooting

### Terraform Errors

**Error: Invalid provider configuration**
```bash
# Solution: Check API token
doctl auth list

# Reinitialize
rm -rf .terraform/
terraform init
```

**Error: Resource already exists**
```bash
# Solution: Import existing resource
terraform import module.droplet.digitalocean_droplet.app_server[0] <droplet-id>
```

**Error: State lock**
```bash
# Solution: Force unlock (if stuck)
terraform force-unlock <lock-id>
```

### Server Connection Issues

```bash
# Can't SSH to droplet
# 1. Check firewall rules
doctl compute firewall list

# 2. Check droplet status
doctl compute droplet get <droplet-id>

# 3. Check SSH key
doctl compute ssh-key list

# 4. Use recovery console
# Login to DigitalOcean web console
```

### Application Not Starting

```bash
# SSH to server
ssh root@<server-ip>

# Check systemd service
systemctl status researcherai
journalctl -u researcherai -n 100

# Check Docker
docker ps
docker-compose logs

# Check environment file
cat /opt/researcherai/.env

# Manual start
cd /opt/researcherai
docker-compose up -d
```

## Security Best Practices

### 1. Secrets Management

```bash
# Never commit secrets to Git!
echo "terraform.tfvars" >> .gitignore
echo "*.tfvars" >> .gitignore

# Use environment variables
export TF_VAR_google_api_key="..."
export TF_VAR_do_token="..."

# Or use secret management service
# - HashiCorp Vault
# - AWS Secrets Manager
# - DigitalOcean Spaces with encryption
```

### 2. SSH Key Rotation

```bash
# Generate new SSH key
ssh-keygen -t ed25519 -f ~/.ssh/researcherai-prod

# Add to DigitalOcean
doctl compute ssh-key import prod-key-2024 \
  --public-key-file ~/.ssh/researcherai-prod.pub

# Update terraform.tfvars
ssh_key_ids = ["12345678", "87654321"]

# Apply
terraform apply

# Remove old key after verification
```

### 3. Firewall Rules

```bash
# Restrict SSH to specific IPs
allowed_ssh_ips = [
  "203.0.113.10/32",  # Office IP
  "198.51.100.50/32", # VPN IP
]

# Monitor firewall logs
doctl compute firewall list
```

### 4. Database Security

```hcl
# Always use private network
private_network_uuid = var.vpc_uuid

# Restrict access with firewall
resource "digitalocean_database_firewall" "postgres_fw" {
  cluster_id = digitalocean_database_cluster.postgres.id

  rule {
    type  = "tag"
    value = var.project_name
  }
}

# Rotate database passwords regularly
```

## Next Steps

Congratulations! You now have:
- ✅ Complete infrastructure as code with Terraform
- ✅ Modular, reusable Terraform modules
- ✅ Multi-environment support (dev/staging/prod)
- ✅ Automated server initialization
- ✅ Load balancing and high availability
- ✅ Managed databases (PostgreSQL, Kafka)
- ✅ Network security with firewall rules
- ✅ Cost optimization strategies

In the next chapter, we'll cover **Observability & Monitoring** including:
- LangFuse and LangSmith for LLM observability
- Prometheus and Grafana for metrics
- Distributed tracing with Jaeger
- Centralized logging with Loki
- Custom dashboards and alerts

Let's ensure we can see what's happening in production!
