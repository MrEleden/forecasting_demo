# Simple PowerShell Deployment Script
param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "stop", "logs")]
    [string]$Action = "deploy"
)

Write-Host "ML Portfolio - Simple Deployment" -ForegroundColor Green

switch ($Action) {
    "deploy" {
        Write-Host "Starting deployment..." -ForegroundColor Yellow
        Set-Location "ci/docker"
        docker-compose up -d
        Write-Host "Application available at: http://localhost:8501" -ForegroundColor Green
    }
    "stop" {
        Write-Host "Stopping services..." -ForegroundColor Yellow
        Set-Location "ci/docker"
        docker-compose down
        Write-Host "Services stopped" -ForegroundColor Green
    }
    "logs" {
        Write-Host "Showing logs..." -ForegroundColor Yellow
        Set-Location "ci/docker"
        docker-compose logs -f
    }
}

# ============================================================================
# Build and Push Images
# ============================================================================
function Build-AndPush {
    Write-Host "Building Docker images..." -ForegroundColor Cyan
    
    # Build multi-stage images
    Write-Host "Building production image..." -ForegroundColor Yellow
    docker build -f ci/docker/Dockerfile `
        --target production `
        -t "${ProjectName}:${ImageTag}" `
        -t "${ProjectName}:latest" .
    
    Write-Host "Building API image..." -ForegroundColor Yellow
    docker build -f ci/docker/Dockerfile `
        --target api `
        -t "${ProjectName}-api:${ImageTag}" `
        -t "${ProjectName}-api:latest" .
    
    # Tag for registry if specified
    if ($DockerRegistry) {
        Write-Host "Tagging images for registry..." -ForegroundColor Yellow
        docker tag "${ProjectName}:${ImageTag}" "${DockerRegistry}/${ProjectName}:${ImageTag}"
        docker tag "${ProjectName}-api:${ImageTag}" "${DockerRegistry}/${ProjectName}-api:${ImageTag}"
        
        Write-Host "Pushing images to registry..." -ForegroundColor Yellow
        docker push "${DockerRegistry}/${ProjectName}:${ImageTag}"
        docker push "${DockerRegistry}/${ProjectName}-api:${ImageTag}"
    }
    
    Write-Host "Image build and push completed" -ForegroundColor Green
}

# ============================================================================
# Deploy Services
# ============================================================================
function Deploy-Services {
    Write-Host "Deploying services..." -ForegroundColor Cyan
    
    # Set environment variables for Docker Compose
    $env:COMPOSE_PROJECT_NAME = "$ProjectName-$Environment"
    $env:IMAGE_TAG = $ImageTag
    
    # Create environment-specific compose file if it doesn't exist
    $envComposeFile = "ci/docker/docker-compose.$Environment.yml"
    if (-not (Test-Path $envComposeFile)) {
        Write-Host "Creating environment-specific compose file..." -ForegroundColor Yellow
        Copy-Item "ci/docker/docker-compose.yml" $envComposeFile
    }
    
    # Deploy using Docker Compose
    Write-Host "Starting Docker Compose deployment..." -ForegroundColor Yellow
    docker-compose -f ci/docker/docker-compose.yml `
                   -f $envComposeFile `
                   up -d --remove-orphans
    
    Write-Host "Services deployed successfully" -ForegroundColor Green
}

# ============================================================================
# Health Check
# ============================================================================
function Test-Health {
    Write-Host "Performing health checks..." -ForegroundColor Cyan
    
    $maxAttempts = 30
    $attempt = 1
    
    # Check main application
    Write-Host "Checking main application health..." -ForegroundColor Yellow
    while ($attempt -le $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Host "Main application is healthy" -ForegroundColor Green
                break
            }
        }
        catch {
            # Continue trying
        }
        
        if ($attempt -eq $maxAttempts) {
            Write-Error "Main application health check failed"
            exit 1
        }
        
        Write-Host "Attempt $attempt/$maxAttempts : Waiting for application..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        $attempt++
    }
    
    # Check API if deployed
    $apiRunning = docker ps --format "table {{.Names}}" | Select-String "api"
    if ($apiRunning) {
        Write-Host "Checking API service health..." -ForegroundColor Yellow
        $attempt = 1
        while ($attempt -le $maxAttempts) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
                if ($response.StatusCode -eq 200) {
                    Write-Host "API service is healthy" -ForegroundColor Green
                    break
                }
            }
            catch {
                # Continue trying
            }
            
            if ($attempt -eq $maxAttempts) {
                Write-Warning "API health check failed"
                break
            }
            
            Write-Host "Attempt $attempt/$maxAttempts : Waiting for API..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
            $attempt++
        }
    }
    
    Write-Host "Health checks completed" -ForegroundColor Green
}

# ============================================================================
# Rollback Function
# ============================================================================
function Invoke-Rollback {
    Write-Host "Rolling back deployment..." -ForegroundColor Cyan
    
    # Stop current deployment
    $envComposeFile = "ci/docker/docker-compose.$Environment.yml"
    docker-compose -f ci/docker/docker-compose.yml `
                   -f $envComposeFile `
                   down
    
    # Deploy previous version if available
    if ($PreviousTag) {
        Write-Host "Deploying previous version: $PreviousTag" -ForegroundColor Yellow
        $env:IMAGE_TAG = $PreviousTag
        Deploy-Services
    }
    else {
        Write-Warning "No previous version specified for rollback"
    }
    
    Write-Host "Rollback completed" -ForegroundColor Green
}

# ============================================================================
# Cleanup Function
# ============================================================================
function Invoke-Cleanup {
    Write-Host "Cleaning up old images and containers..." -ForegroundColor Cyan
    
    # Remove old images (keep last 3 versions)
    try {
        $oldImages = docker images $ProjectName --format "table {{.Repository}}:{{.Tag}}" | 
                    Where-Object { $_ -notmatch "latest" } | 
                    Select-Object -Skip 3
        
        foreach ($image in $oldImages) {
            Write-Host "Removing old image: $image" -ForegroundColor Yellow
            docker rmi $image -f
        }
    }
    catch {
        Write-Warning "Could not remove some old images"
    }
    
    # Remove unused containers and networks
    Write-Host "Pruning unused Docker resources..." -ForegroundColor Yellow
    docker system prune -f
    
    Write-Host "Cleanup completed" -ForegroundColor Green
}

# ============================================================================
# Main Execution
# ============================================================================
switch ($Action) {
    "validate" {
        Validate-Environment
    }
    "build" {
        Validate-Environment
        Build-AndPush
    }
    "deploy" {
        Validate-Environment
        Build-AndPush
        Deploy-Services
        Test-Health
    }
    "health" {
        Test-Health
    }
    "rollback" {
        Invoke-Rollback
    }
    "cleanup" {
        Invoke-Cleanup
    }
    default {
        Write-Host "Usage: .\deploy.ps1 {validate|build|deploy|health|rollback|cleanup}" -ForegroundColor Red
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Yellow
        Write-Host "  validate  - Validate deployment environment"
        Write-Host "  build     - Build and push Docker images"
        Write-Host "  deploy    - Full deployment (build + deploy + health check)"
        Write-Host "  health    - Run health checks on deployed services"
        Write-Host "  rollback  - Rollback to previous version"
        Write-Host "  cleanup   - Clean up old images and containers"
        Write-Host ""
        Write-Host "Parameters:" -ForegroundColor Yellow
        Write-Host "  -Environment    : Target environment (default: staging)"
        Write-Host "  -ImageTag       : Docker image tag (default: latest)"
        Write-Host "  -DockerRegistry : Docker registry URL"
        Write-Host "  -PreviousTag    : Previous image tag for rollback"
        exit 1
    }
}

Write-Host "Deployment script completed successfully" -ForegroundColor Green