# ===============================
# JPetStore Container Auto Start Script
# ===============================

# Set variables
$DB_DIR = "$PSScriptRoot\db_data"
$CONTAINER_NAME = "jpetstore_app"
$IMAGE_NAME = "jpetstore:latest"
$HOST_PORT = 8082
$CONTAINER_PORT = 8080

Write-Host "🔍 JPetStore Auto Start Script" -ForegroundColor Green

# Create local database directory
if (-Not (Test-Path $DB_DIR)) {
    Write-Host "📁 Creating database directory: $DB_DIR" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $DB_DIR | Out-Null
} else {
    Write-Host "📁 Database directory exists: $DB_DIR" -ForegroundColor Green
}

# Check and cleanup existing container
$existingContainer = docker ps -aq -f "name=$CONTAINER_NAME"
if ($existingContainer) {
    Write-Host "⚠️ Stopping and removing existing container: $CONTAINER_NAME" -ForegroundColor Yellow
    docker stop $CONTAINER_NAME | Out-Null
    docker rm $CONTAINER_NAME | Out-Null
}

# Start container
Write-Host "🚀 Starting container: $CONTAINER_NAME" -ForegroundColor Green
Write-Host "   Port mapping: $HOST_PORT -> $CONTAINER_PORT"
Write-Host "   Data directory: $DB_DIR"

$dockerCommand = "docker run -d --name $CONTAINER_NAME -p ${HOST_PORT}:${CONTAINER_PORT} -v `"${DB_DIR}:/usr/local/tomcat/db`" $IMAGE_NAME"
Write-Host "Executing: $dockerCommand" -ForegroundColor Cyan

try {
    $containerId = Invoke-Expression $dockerCommand
    Write-Host "✅ Container started successfully, ID: $containerId" -ForegroundColor Green
} catch {
    Write-Host "❌ Container start failed: $_" -ForegroundColor Red
    exit 1
}

# Wait for container to start
Write-Host "⏳ Waiting for container to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check container status
Write-Host "📊 Container Status:" -ForegroundColor Cyan
docker ps -f "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Display access information
Write-Host ""
Write-Host "🎯 JPetStore is ready!" -ForegroundColor Green
Write-Host "   Access URL: http://localhost:$HOST_PORT" -ForegroundColor Cyan
Write-Host "   Data directory: $DB_DIR" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Useful commands:" -ForegroundColor Yellow
Write-Host "   View logs: docker logs -f $CONTAINER_NAME"
Write-Host "   Stop container: docker stop $CONTAINER_NAME"
Write-Host "   Enter container: docker exec -it $CONTAINER_NAME bash"
