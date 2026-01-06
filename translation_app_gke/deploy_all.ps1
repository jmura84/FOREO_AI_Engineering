$ProjectId = "gen-lang-client-0088002625"
$ErrorActionPreference = "Stop"

Write-Host "Starting Build and Push Process..." -ForegroundColor Cyan
# Build and Push the main app image (contains Gemini logic + Whisper)
.\build_and_push.ps1 -ProjectId $ProjectId

if ($LASTEXITCODE -eq 0) {
    Write-Host "Applying Kubernetes manifests..." -ForegroundColor Cyan
    
    # We are skipping local vLLM models (01 and 02) since we use Gemini API + local Whisper in the app container.
    # kubectl apply -f k8s/01-vllm-text.yaml
    # kubectl apply -f k8s/02-vllm-ocr.yaml
    
    # Apply the main application deployment
    kubectl apply -f k8s/03-app.yaml

    Write-Host "Restarting deployment to pull latest image..." -ForegroundColor Cyan
    kubectl rollout restart deployment translation-app
    
    Write-Host "Deployment updated. Monitoring rollout status..." -ForegroundColor Cyan
    kubectl rollout status deployment/translation-app
    
    Write-Host "Deployment successfully updated!" -ForegroundColor Green
    
    $service = kubectl get svc translation-app-service -o json | ConvertFrom-Json
    if ($service.status.loadBalancer.ingress) {
        $ip = $service.status.loadBalancer.ingress[0].ip
        Write-Host "Application is available at: http://$ip" -ForegroundColor Green
    }
    else {
        Write-Host "External IP is still pending. Run 'kubectl get svc translation-app-service' later to check." -ForegroundColor Yellow
    }

}
else {
    Write-Host "Build failed, aborting deployment." -ForegroundColor Red
    exit 1
}
