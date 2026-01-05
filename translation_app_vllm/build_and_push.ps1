param(
    [Parameter(Mandatory = $true, HelpMessage = "gen-lang-client-0088002625")]
    [string]$ProjectId,

    [string]$ImageName = "translation-app-vllm",

    [string]$Tag = "latest",
    
    [string]$RegistryHost = "gcr.io"
)

$ErrorActionPreference = "Stop"

$FullImageName = "$RegistryHost/$ProjectId/$ImageName`:$Tag"

Write-Host "Building Docker image: $FullImageName" -ForegroundColor Cyan
docker build -t $FullImageName .

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful. Pushing to registry..." -ForegroundColor Cyan
    docker push $FullImageName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully pushed $FullImageName" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to push image. Please ensure you are authenticated with gcloud auth configure-docker" -ForegroundColor Red
    }
}
else {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}
