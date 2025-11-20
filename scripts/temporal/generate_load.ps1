# PowerShell 脚本：对四个单体应用打流量，产生 OTel trace/log/metrics
# 需要本地安装 Apache Benchmark (ab.exe)

# 参数设置
$requests = 500      # 总请求数
$concurrency = 20    # 并发数

# 应用 URL 列表
$apps = @(
    "http://localhost:8081/",        # Acmeair
    "http://localhost:8082/",        # Daytrader7
    "http://localhost:8083/jpetstore/", # JPetStore
    "http://localhost:8084/"         # PlantsByWebSphere
)

Write-Host "开始对四个单体应用进行流量测试..."
foreach ($url in $apps) {
    Write-Host ">>> 压测 $url ..."
    & ab -n $requests -c $concurrency $url
    Start-Sleep -Seconds 5  # 每个应用之间停 5 秒，防止打爆环境
}
Write-Host "流量生成完成，请检查 OTel/Jaeger 收集到的数据。"
