# Docker Deployment Guide

## Quick Start

### 1. Build and Run All Services
```bash
docker-compose up --build
```

This starts:
- **API Service**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

### 2. Run Individual Services

#### API Only
```bash
docker-compose up api
```

#### Training Only
```bash
docker-compose run training
```

#### Dashboard Only
```bash
docker-compose up dashboard
```

## Available Docker Images

### 1. Serving (Production API)
```bash
docker build --target serving -t forecasting-api .
docker run -p 8000:8000 forecasting-api
```

**Features**:
- Optimized for production
- Gunicorn + Uvicorn workers
- Non-root user for security
- Health checks enabled
- Multi-worker support (4 workers)

### 2. Training
```bash
docker build --target training -t forecasting-training .
docker run -v $(pwd)/data:/app/data forecasting-training
```

**Features**:
- Pre-configured for model training
- MLflow tracking enabled
- Volume mounts for data and models

### 3. Development
```bash
docker build --target development -t forecasting-dev .
docker run -it -p 8888:8888 forecasting-dev
```

**Features**:
- Jupyter Lab included
- All dev tools installed
- Tests and documentation included

### 4. Dashboard
```bash
docker-compose up dashboard
```

Access at: http://localhost:8501

## Configuration

### Environment Variables

```bash
# API Service
MLFLOW_TRACKING_URI=/app/mlruns
MODEL_DIR=/app/models

# Training Service
HYDRA_FULL_ERROR=1
PYTHONPATH=/app/src
```

### Volume Mounts

```yaml
volumes:
  - ./data:/app/data              # Training data
  - ./models:/app/models          # Saved models
  - ./mlruns:/app/mlruns          # MLflow tracking
  - ./results:/app/results        # Benchmark results
```

## Production Deployment

### 1. Build Production Image
```bash
docker build --target serving -t forecasting-api:v1.0 .
```

### 2. Push to Registry
```bash
docker tag forecasting-api:v1.0 your-registry/forecasting-api:v1.0
docker push your-registry/forecasting-api:v1.0
```

### 3. Deploy with Docker Compose
```bash
docker-compose -f docker-compose.yml up -d api mlflow
```

### 4. Scale API Workers
```bash
docker-compose up -d --scale api=3
```

## Health Checks

### API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": 2
}
```

### Docker Health Status
```bash
docker ps --filter "health=healthy"
docker inspect forecasting_api | grep -A 5 Health
```

## Troubleshooting

### View Logs
```bash
docker-compose logs api
docker-compose logs -f training
```

### Enter Container
```bash
docker exec -it forecasting_api bash
```

### Rebuild Without Cache
```bash
docker-compose build --no-cache
```

### Remove All Containers and Volumes
```bash
docker-compose down -v
```

## Performance Optimization

### 1. Multi-stage Builds
The Dockerfile uses multi-stage builds to minimize image size:
- Base: ~500MB
- Serving: ~800MB (optimized)
- Development: ~1.2GB (includes all tools)

### 2. Caching
Dependencies are cached in separate layers for faster rebuilds.

### 3. Resource Limits
Add to docker-compose.yml:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Security

### 1. Non-root User
Serving image runs as non-root user `mluser`.

### 2. Read-only Filesystem (Optional)
```yaml
services:
  api:
    read_only: true
    tmpfs:
      - /tmp
```

### 3. Secrets Management
Use Docker secrets for sensitive data:
```bash
echo "secret_key" | docker secret create api_key -
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Build Docker Image
  run: docker build --target serving -t ${{ secrets.DOCKER_REGISTRY }}/forecasting-api:${{ github.sha }} .

- name: Push to Registry
  run: docker push ${{ secrets.DOCKER_REGISTRY }}/forecasting-api:${{ github.sha }}
```

## Monitoring

### Container Stats
```bash
docker stats forecasting_api
```

### Resource Usage
```bash
docker-compose top
```

### Logs Aggregation
Consider using:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Prometheus + Grafana
- Docker logging drivers (json-file, syslog, fluentd)

## Next Steps

1. Configure persistent volumes for production data
2. Set up load balancer (nginx, traefik)
3. Implement container orchestration (Kubernetes, Docker Swarm)
4. Add monitoring and alerting
5. Configure automated backups
