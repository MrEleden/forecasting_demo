# Simple CI/CD for ML Portfolio

This directory contains simple CI/CD infrastructure for the ML forecasting portfolio project.

## Structure

```
ci/
├── README.md                  # This documentation
├── github-actions/            # GitHub Actions workflows
│   ├── cicd-pipeline.yml     # Simple CI/CD pipeline
│   └── validate-structure.yml # Project structure validation
├── scripts/                   # Deployment scripts
│   ├── validate_structure.py  # Structure validation
│   ├── deploy.sh              # Linux/macOS deployment
│   ├── deploy.ps1             # Windows deployment
│   └── setup_cicd.py          # Setup script
└── docker/                    # Docker configuration
    ├── Dockerfile             # Simple container build
    └── docker-compose.yml     # Single service
```

## Quick Start

### Local Deployment

```bash
# Linux/macOS
./ci/scripts/deploy.sh deploy

# Windows
.\ci\scripts\deploy.ps1 deploy

# Access application
# http://localhost:8501
```

### Setup CI/CD

```bash
# Copy workflows to GitHub
python ci/scripts/setup_cicd.py

# Commit and push
git add .github/workflows/
git commit -m "Add simple CI/CD"
git push
```

## Features

### Simple CI Pipeline
- **Code Quality**: Basic linting with ruff and black
- **Structure Validation**: Project folder structure checks
- **Testing**: Basic pytest execution
- **Fast**: Minimal dependencies and quick execution

### Simple Deployment
- **Docker**: Single container deployment
- **Local**: Easy local development setup
- **Cross-Platform**: Works on Windows, macOS, Linux

### Easy Management
- **Three Commands**: deploy, stop, logs
- **No Complexity**: No multiple environments or advanced features
- **Quick Setup**: One command to get started

## Commands

### Deployment Scripts

```bash
# Deploy application
./ci/scripts/deploy.sh deploy        # Linux/macOS
.\ci\scripts\deploy.ps1 deploy       # Windows

# Stop application
./ci/scripts/deploy.sh stop          # Linux/macOS
.\ci\scripts\deploy.ps1 stop         # Windows

# View logs
./ci/scripts/deploy.sh logs          # Linux/macOS
.\ci\scripts\deploy.ps1 logs         # Windows
```

### Validation

```bash
# Validate project structure
python ci/scripts/validate_structure.py

# Fix missing folders
python ci/scripts/validate_structure.py --fix
```

## Docker

### Simple Container
- **Single Stage**: No complex multi-stage builds
- **Essential Dependencies**: Only what's needed to run
- **Health Check**: Basic application health monitoring
- **Non-Root**: Runs as non-root user for security

### Single Service
- **Streamlit App**: Port 8501
- **Data Volume**: Read-only data access
- **Auto-Restart**: Restarts on failure

## GitHub Actions

### CI/CD Pipeline
- **Test**: Code quality and structure validation
- **Deploy**: Simple deployment indication
- **Triggers**: Push to main, pull requests
- **Fast**: Completes in under 5 minutes

### Structure Validation
- **Consistency**: Ensures all projects follow same structure
- **Auto-Fix**: Can automatically create missing folders
- **Cross-Platform**: Works on all operating systems

## Troubleshooting

### Common Issues

1. **Docker not found**:
   ```bash
   # Install Docker Desktop
   # https://www.docker.com/products/docker-desktop/
   ```

2. **Port already in use**:
   ```bash
   # Stop existing containers
   docker stop $(docker ps -q)
   ```

3. **Permission errors**:
   ```bash
   # Linux/macOS - make scripts executable
   chmod +x ci/scripts/deploy.sh
   ```

### Logs and Debugging

```bash
# View application logs
docker logs ml-app

# Check container status
docker ps -a

# Remove all containers
docker system prune -a
```

## Next Steps

Once you're comfortable with this simple setup, you can:

1. **Add More Services**: Database, API, monitoring
2. **Advanced CI**: Security scanning, performance testing
3. **Multiple Environments**: Staging, production deployments
4. **Monitoring**: Prometheus, Grafana integration

---

*This simple CI/CD setup gets you started quickly without overwhelming complexity. Perfect for development and learning.*

### GitHub Actions Setup
The workflow files in `github-actions/` are templates that should be symlinked or copied to `.github/workflows/` for GitHub to recognize them.

### Local Development
CI scripts can be run locally to validate changes before committing:
```bash
# Validate project structure
python ci/scripts/validate_structure.py

# Fix any structural issues
python ci/scripts/validate_structure.py --fix
```

## 🔄 Maintenance

- **Structure Validator**: Update `REQUIRED_STRUCTURE` in `scripts/validate_structure.py` when adding new mandatory folders
- **Workflows**: Modify workflow files to add new validation steps or change triggers
- **Documentation**: Keep this README updated when adding new CI/CD components

---

*This CI/CD infrastructure ensures professional-grade code quality and structural consistency across the ML portfolio.*