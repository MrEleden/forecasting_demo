#!/bin/bash
# Simple Deployment Script for ML Portfolio

set -e

ACTION=${1:-deploy}

echo "ML Portfolio - Simple Deployment"

case "$ACTION" in
    "deploy")
        echo "Starting deployment..."
        cd ci/docker
        docker-compose up -d
        echo "Application available at: http://localhost:8501"
        ;;
    "stop")
        echo "Stopping services..."
        cd ci/docker
        docker-compose down
        echo "Services stopped"
        ;;
    "logs")
        echo "Showing logs..."
        cd ci/docker
        docker-compose logs -f
        ;;
    *)
        echo "Usage: $0 {deploy|stop|logs}"
        exit 1
        ;;
esac

# ============================================================================
# Build and Push Images
# ============================================================================
build_and_push() {
    echo "Building Docker images..."
    
    # Build multi-stage images
    docker build -f ci/docker/Dockerfile \
        --target production \
        -t ${PROJECT_NAME}:${IMAGE_TAG} \
        -t ${PROJECT_NAME}:latest .
    
    docker build -f ci/docker/Dockerfile \
        --target api \
        -t ${PROJECT_NAME}-api:${IMAGE_TAG} \
        -t ${PROJECT_NAME}-api:latest .
    
    # Tag for registry if specified
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        docker tag ${PROJECT_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}
        docker tag ${PROJECT_NAME}-api:${IMAGE_TAG} ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:${IMAGE_TAG}
        
        echo "Pushing images to registry..."
        docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}
        docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:${IMAGE_TAG}
    fi
    
    echo "Image build and push completed"
}

# ============================================================================
# Deploy Services
# ============================================================================
deploy_services() {
    echo "Deploying services..."
    
    # Set environment-specific configurations
    export COMPOSE_PROJECT_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
    export IMAGE_TAG="${IMAGE_TAG}"
    
    # Create environment-specific compose file if it doesn't exist
    if [[ ! -f "ci/docker/docker-compose.${ENVIRONMENT}.yml" ]]; then
        echo "Creating environment-specific compose file..."
        cp ci/docker/docker-compose.yml ci/docker/docker-compose.${ENVIRONMENT}.yml
    fi
    
    # Deploy using Docker Compose
    docker-compose -f ci/docker/docker-compose.yml \
                   -f ci/docker/docker-compose.${ENVIRONMENT}.yml \
                   up -d --remove-orphans
    
    echo "Services deployed successfully"
}

# ============================================================================
# Health Check
# ============================================================================
health_check() {
    echo "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    # Check main application
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
            echo "Main application is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            echo "ERROR: Main application health check failed"
            exit 1
        fi
        
        echo "Attempt $attempt/$max_attempts: Waiting for application..."
        sleep 10
        ((attempt++))
    done
    
    # Check API if deployed
    if docker ps --format "table {{.Names}}" | grep -q "api"; then
        attempt=1
        while [[ $attempt -le $max_attempts ]]; do
            if curl -f http://localhost:8000/health &>/dev/null; then
                echo "API service is healthy"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                echo "WARNING: API health check failed"
            fi
            
            echo "Attempt $attempt/$max_attempts: Waiting for API..."
            sleep 5
            ((attempt++))
        done
    fi
    
    echo "Health checks completed"
}

# ============================================================================
# Rollback Function
# ============================================================================
rollback() {
    echo "Rolling back deployment..."
    
    # Stop current deployment
    docker-compose -f ci/docker/docker-compose.yml \
                   -f ci/docker/docker-compose.${ENVIRONMENT}.yml \
                   down
    
    # Deploy previous version (if available)
    if [[ -n "$PREVIOUS_TAG" ]]; then
        export IMAGE_TAG="$PREVIOUS_TAG"
        deploy_services
    else
        echo "No previous version specified for rollback"
    fi
    
    echo "Rollback completed"
}

# ============================================================================
# Cleanup Function
# ============================================================================
cleanup() {
    echo "Cleaning up old images and containers..."
    
    # Remove old images (keep last 3 versions)
    docker images ${PROJECT_NAME} --format "table {{.Repository}}:{{.Tag}}" | \
        grep -v "latest" | tail -n +4 | xargs -r docker rmi || true
    
    # Remove unused containers and networks
    docker system prune -f
    
    echo "Cleanup completed"
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    case "${1:-deploy}" in
        "validate")
            validate_environment
            ;;
        "build")
            validate_environment
            build_and_push
            ;;
        "deploy")
            validate_environment
            build_and_push
            deploy_services
            health_check
            ;;
        "health")
            health_check
            ;;
        "rollback")
            rollback
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 {validate|build|deploy|health|rollback|cleanup}"
            echo ""
            echo "Commands:"
            echo "  validate  - Validate deployment environment"
            echo "  build     - Build and push Docker images"
            echo "  deploy    - Full deployment (build + deploy + health check)"
            echo "  health    - Run health checks on deployed services"
            echo "  rollback  - Rollback to previous version"
            echo "  cleanup   - Clean up old images and containers"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"