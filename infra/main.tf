provider "aws" {
  region = "us-east-1"
}

# 1. ECR Repository (Where Docker images live)
resource "aws_ecr_repository" "api_repo" {
  name = "bully-api"
}

# 2. ECS Cluster (The "Serverless" Fleet)
resource "aws_ecs_cluster" "main" {
  name = "bully-cluster"
}

# 3. IAM Role (Permissions for the task)
resource "aws_iam_role" "ecs_execution_role" {
  name = "bully_ecs_execution_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" } }]
  })
}

# Attach standard policies so Fargate can pull images and log to CloudWatch
resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# 4. Task Definition (The "Blueprint")
resource "aws_ecs_task_definition" "api" {
  family                   = "bully-api-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512 # 0.5 vCPU
  memory                   = 1024 # 1 GB
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([{
    name  = "api-container"
    image = "${aws_ecr_repository.api_repo.repository_url}:latest"
    portMappings = [{ containerPort = 8000 }]
    environment = [
      { name = "MLFLOW_TRACKING_URI", value = "YOUR_DAGSHUB_URL" },
      { name = "MODEL_STAGE", value = "Production" }
    ],
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/bully-api"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

# 5. CloudWatch Logs
resource "aws_cloudwatch_log_group" "logs" {
  name = "/ecs/bully-api"
}

# 6. Service (Ensures 1 copy is always running)
resource "aws_ecs_service" "api_service" {
  name            = "bully-api-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = ["subnet-xxxxxx", "subnet-yyyyyy"] # Replace with your Default VPC Subnet IDs
    security_groups  = [aws_security_group.allow_http.id]
    assign_public_ip = true
  }
}

# Security Group (Allow Traffic on Port 8000)
resource "aws_security_group" "allow_http" {
  name        = "bully-api-sg"
  description = "Allow inbound traffic on 8000"

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}