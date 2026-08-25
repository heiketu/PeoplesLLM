Set-StrictMode -Version 2.0

function New-LauncherProfile {
    [ordered]@{
        Version                  = 1
        ServerPath               = ".\build\bin\Release\llama-server.exe"
        ModelPath                = ""
        ModelAlias               = "local-model"
        DSparkPath               = ""
        Host                     = "127.0.0.1"
        Port                     = 8080
        ContextSize              = 8192
        Slots                    = 1
        Threads                  = 0
        ThreadsBatch             = 0
        BatchSize                = 2048
        UBatchSize               = 512
        Device                   = "none"
        GpuLayers                = "0"
        CpuMoeLayers             = 0
        SplitMode                = "none"
        TensorSplit              = ""
        MainGpu                  = 0
        Numa                     = "none"
        NumaMirror               = "all"
        LoadMode                 = "auto"
        CacheTypeK               = "q8_0"
        CacheTypeV               = "q8_0"
        FlashAttention           = "auto"
        NoWarmup                 = $false
        DSparkNMax               = 3
        DSparkPMin               = 0.0
        DraftDevice              = ""
        DraftGpuLayers           = "auto"
        EnableNumaEp             = $false
        EnableHierBarrier        = $true
        EnableGateUpParallel     = $false
        EnableRemoteEp           = $false
        RemoteEndpoints          = ""
        RemoteLayers             = "0-42"
        RemoteKLocal             = 0
        RemoteMaxEffort          = $false
        RemotePp                 = $true
        RemoteRdma               = $false
        RemoteWeightOnMaster     = $true
        RemoteParallelIo         = $true
        RemoteReconnectTimeoutMs = 90000
        ExtraArguments           = @()
        ExtraEnvironment         = ""
    }
}

function Merge-LauncherProfile {
    param([Parameter(Mandatory = $true)]$Profile)
    $result = New-LauncherProfile
    if ($null -eq $Profile) {
        return $result
    }
    if ($Profile -is [Collections.IDictionary]) {
        foreach ($key in $Profile.Keys) {
            if ($result.Contains([string]$key)) { $result[[string]$key] = $Profile[$key] }
        }
    } else {
        foreach ($property in $Profile.PSObject.Properties) {
            if ($result.Contains($property.Name)) { $result[$property.Name] = $property.Value }
        }
    }
    return $result
}

function ConvertTo-RequiredInt {
    param([string]$Name, $Value, [int]$Minimum, [int]$Maximum)
    $parsed = 0
    if (-not [int]::TryParse([string]$Value, [ref]$parsed) -or $parsed -lt $Minimum -or $parsed -gt $Maximum) {
        throw "$Name must be an integer in [$Minimum, $Maximum]"
    }
    return $parsed
}

function Test-LauncherProfile {
    param(
        [Parameter(Mandatory = $true)]$Profile,
        [switch]$RequireFiles
    )
    $profile = Merge-LauncherProfile $Profile
    $errors = New-Object System.Collections.Generic.List[string]
    if ([string]::IsNullOrWhiteSpace([string]$profile.ServerPath)) { $errors.Add("ServerPath is required") }
    if ([string]::IsNullOrWhiteSpace([string]$profile.ModelPath)) { $errors.Add("ModelPath is required") }
    if ([string]::IsNullOrWhiteSpace([string]$profile.ModelAlias)) { $errors.Add("ModelAlias is required") }
    if ([string]::IsNullOrWhiteSpace([string]$profile.Host)) { $errors.Add("Host is required") }
    foreach ($check in @(
        @("Port", $profile.Port, 1, 65535),
        @("ContextSize", $profile.ContextSize, 0, 2147483647),
        @("Slots", $profile.Slots, 1, 4096),
        @("Threads", $profile.Threads, 0, 4096),
        @("ThreadsBatch", $profile.ThreadsBatch, 0, 4096),
        @("BatchSize", $profile.BatchSize, 1, 1048576),
        @("UBatchSize", $profile.UBatchSize, 1, 1048576),
        @("CpuMoeLayers", $profile.CpuMoeLayers, 0, 100000),
        @("MainGpu", $profile.MainGpu, 0, 1024),
        @("DSparkNMax", $profile.DSparkNMax, 0, 1024),
        @("RemoteKLocal", $profile.RemoteKLocal, 0, 1024),
        @("RemoteReconnectTimeoutMs", $profile.RemoteReconnectTimeoutMs, 0, 300000)
    )) {
        try { [void](ConvertTo-RequiredInt $check[0] $check[1] $check[2] $check[3]) } catch { $errors.Add($_.Exception.Message) }
    }
    $pmin = 0.0
    if (-not [double]::TryParse([string]$profile.DSparkPMin, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$pmin) -or $pmin -lt 0.0 -or $pmin -gt 1.0) {
        $errors.Add("DSparkPMin must be in [0, 1]")
    }
    if (@("none", "layer", "row", "tensor") -notcontains [string]$profile.SplitMode) { $errors.Add("SplitMode is invalid") }
    if (@("none", "distribute", "isolate", "numactl", "mirror") -notcontains [string]$profile.Numa) { $errors.Add("Numa is invalid") }
    if (@("auto", "none", "mmap", "mlock", "mmap+mlock", "dio") -notcontains [string]$profile.LoadMode) { $errors.Add("LoadMode is invalid") }
    if (@("auto", "on", "off") -notcontains [string]$profile.FlashAttention) { $errors.Add("FlashAttention is invalid") }
    foreach ($name in @("NoWarmup", "EnableNumaEp", "EnableHierBarrier", "EnableGateUpParallel", "EnableRemoteEp", "RemoteMaxEffort", "RemotePp", "RemoteRdma", "RemoteWeightOnMaster", "RemoteParallelIo")) {
        if ($profile[$name] -isnot [bool]) { $errors.Add("$name must be a JSON boolean") }
    }
    foreach ($name in @("GpuLayers", "DraftGpuLayers")) {
        if ([string]($profile[$name]) -notmatch '^(auto|all|[0-9]+)$') { $errors.Add("$name must be auto, all, or a non-negative integer") }
    }
    $cacheTypes = @("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
    if ($cacheTypes -notcontains [string]$profile.CacheTypeK) { $errors.Add("CacheTypeK is invalid") }
    if ($cacheTypes -notcontains [string]$profile.CacheTypeV) { $errors.Add("CacheTypeV is invalid") }
    if ($profile.EnableRemoteEp -and [string]::IsNullOrWhiteSpace([string]$profile.RemoteEndpoints)) { $errors.Add("RemoteEndpoints is required when remote EP is enabled") }
    if ($profile.EnableRemoteEp -and [string]::IsNullOrWhiteSpace([string]$profile.RemoteLayers)) { $errors.Add("RemoteLayers is required when remote EP is enabled") }
    if ($profile.RemoteMaxEffort -and [string]$profile.RemoteKLocal -ne "0") { $errors.Add("RemoteMaxEffort requires RemoteKLocal=0") }
    if ($profile.EnableNumaEp -and (@("distribute", "mirror") -notcontains [string]$profile.Numa)) { $errors.Add("CPU NUMA EP requires Numa=distribute or mirror") }
    if ($profile.EnableNumaEp -and (@("auto", "mmap", "mmap+mlock") -contains [string]$profile.LoadMode)) { $errors.Add("CPU NUMA EP requires a non-mmap load mode such as none") }
    if (-not [string]::IsNullOrWhiteSpace([string]$profile.DSparkPath) -and [string]$profile.DSparkPath -eq [string]$profile.ModelPath) { $errors.Add("DSparkPath must not equal ModelPath") }
    if ($RequireFiles) {
        if (-not [IO.File]::Exists([string]$profile.ServerPath)) { $errors.Add("ServerPath does not exist") }
        if (-not [IO.File]::Exists([string]$profile.ModelPath)) { $errors.Add("ModelPath does not exist") }
        if (-not [string]::IsNullOrWhiteSpace([string]$profile.DSparkPath) -and -not [IO.File]::Exists([string]$profile.DSparkPath)) { $errors.Add("DSparkPath does not exist") }
    }
    foreach ($line in ([string]$profile.ExtraEnvironment -split "\r?\n")) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line -notmatch '^[A-Za-z_][A-Za-z0-9_]*=.*$') { $errors.Add("Invalid environment line: $line") }
    }
    foreach ($argument in @($profile.ExtraArguments)) {
        if ([string]$argument -match '[\r\n]') { $errors.Add("ExtraArguments entries must not contain newlines") }
    }
    return $errors.ToArray()
}

function Add-ArgumentPair {
    param([System.Collections.Generic.List[string]]$Arguments, [string]$Name, $Value)
    $Arguments.Add($Name)
    $Arguments.Add([string]$Value)
}

function ConvertTo-LauncherCommand {
    param([Parameter(Mandatory = $true)]$Profile)
    $profile = Merge-LauncherProfile $Profile
    $arguments = New-Object System.Collections.Generic.List[string]
    Add-ArgumentPair $arguments "--model" $profile.ModelPath
    Add-ArgumentPair $arguments "--alias" $profile.ModelAlias
    Add-ArgumentPair $arguments "--host" $profile.Host
    Add-ArgumentPair $arguments "--port" $profile.Port
    Add-ArgumentPair $arguments "--ctx-size" $profile.ContextSize
    Add-ArgumentPair $arguments "--parallel" $profile.Slots
    if ([int]$profile.Threads -gt 0) { Add-ArgumentPair $arguments "--threads" $profile.Threads }
    if ([int]$profile.ThreadsBatch -gt 0) { Add-ArgumentPair $arguments "--threads-batch" $profile.ThreadsBatch }
    Add-ArgumentPair $arguments "--batch-size" $profile.BatchSize
    Add-ArgumentPair $arguments "--ubatch-size" $profile.UBatchSize
    Add-ArgumentPair $arguments "--device" $profile.Device
    Add-ArgumentPair $arguments "--n-gpu-layers" $profile.GpuLayers
    Add-ArgumentPair $arguments "--n-cpu-moe" $profile.CpuMoeLayers
    Add-ArgumentPair $arguments "--split-mode" $profile.SplitMode
    if (-not [string]::IsNullOrWhiteSpace([string]$profile.TensorSplit)) { Add-ArgumentPair $arguments "--tensor-split" $profile.TensorSplit }
    Add-ArgumentPair $arguments "--main-gpu" $profile.MainGpu
    if ([string]$profile.Numa -ne "none") {
        Add-ArgumentPair $arguments "--numa" $profile.Numa
        if ([string]$profile.Numa -eq "mirror") { Add-ArgumentPair $arguments "--numa-mirror" $profile.NumaMirror }
    }
    Add-ArgumentPair $arguments "--load-mode" $profile.LoadMode
    Add-ArgumentPair $arguments "--cache-type-k" $profile.CacheTypeK
    Add-ArgumentPair $arguments "--cache-type-v" $profile.CacheTypeV
    if ([string]$profile.FlashAttention -ne "auto") { Add-ArgumentPair $arguments "--flash-attn" $profile.FlashAttention }
    if ([bool]$profile.NoWarmup) { $arguments.Add("--no-warmup") }
    if (-not [string]::IsNullOrWhiteSpace([string]$profile.DSparkPath)) {
        $draftPMin = [double]::Parse([string]$profile.DSparkPMin, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture)
        Add-ArgumentPair $arguments "--model-draft" $profile.DSparkPath
        Add-ArgumentPair $arguments "--spec-type" "draft-dspark"
        Add-ArgumentPair $arguments "--spec-draft-n-max" $profile.DSparkNMax
        Add-ArgumentPair $arguments "--spec-draft-p-min" ($draftPMin.ToString([Globalization.CultureInfo]::InvariantCulture))
        if (-not [string]::IsNullOrWhiteSpace([string]$profile.DraftDevice)) { Add-ArgumentPair $arguments "--spec-draft-device" $profile.DraftDevice }
        Add-ArgumentPair $arguments "--spec-draft-ngl" $profile.DraftGpuLayers
    }
    foreach ($argument in @($profile.ExtraArguments)) {
        if ($null -ne $argument -and -not [string]::IsNullOrWhiteSpace([string]$argument)) { $arguments.Add([string]$argument) }
    }

    $environment = [ordered]@{}
    if ([bool]$profile.EnableNumaEp) {
        $environment["GGML_NUMA_EP"] = "1"
        if ([bool]$profile.EnableHierBarrier) { $environment["GGML_NUMA_HIER_BARRIER"] = "1" }
        if ([bool]$profile.EnableGateUpParallel) { $environment["GGML_NUMA_EP_GATE_UP_PARALLEL"] = "1" }
    }
    if ([bool]$profile.EnableRemoteEp) {
        $environment["GGML_REMOTE_EP"] = "1"
        $environment["GGML_REMOTE_EP_LAYERS"] = [string]$profile.RemoteLayers
        $environment["GGML_REMOTE_EP_SCHED"] = "1"
        $environment["GGML_REMOTE_EP_SCHED_ENDPOINTS"] = [string]$profile.RemoteEndpoints
        $environment["GGML_REMOTE_EP_SCHED_KLOCAL"] = [string]$profile.RemoteKLocal
        $environment["GGML_REMOTE_EP_SCHED_MAX_EFFORT"] = $(if ([bool]$profile.RemoteMaxEffort) { "1" } else { "0" })
        $environment["GGML_REMOTE_EP_SCHED_PP"] = $(if ([bool]$profile.RemotePp) { "1" } else { "0" })
        $environment["GGML_REMOTE_EP_RDMA"] = $(if ([bool]$profile.RemoteRdma) { "1" } else { "0" })
        $environment["GGML_REMOTE_EP_WEIGHT_ON_MASTER"] = $(if ([bool]$profile.RemoteWeightOnMaster) { "1" } else { "0" })
        $environment["GGML_REMOTE_EP_PARALLEL_IO"] = $(if ([bool]$profile.RemoteParallelIo) { "1" } else { "0" })
        $environment["GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS"] = [string]$profile.RemoteReconnectTimeoutMs
    }
    foreach ($line in ([string]$profile.ExtraEnvironment -split "\r?\n")) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $separator = $line.IndexOf('=')
        if ($separator -le 0) { throw "Invalid environment line: $line" }
        $name = $line.Substring(0, $separator)
        $value = $line.Substring($separator + 1)
        if ($name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$' -or $value.Contains("`r") -or $value.Contains("`n")) { throw "Invalid environment line: $line" }
        if ($environment.Contains($name)) { throw "ExtraEnvironment duplicates generated variable: $name" }
        $environment[$name] = $value
    }
    [pscustomobject]@{
        FileName    = [string]$profile.ServerPath
        Arguments   = $arguments.ToArray()
        Environment = $environment
    }
}

function Quote-WindowsArgument {
    param([AllowEmptyString()][string]$Argument)
    if ($null -eq $Argument -or $Argument.Length -eq 0) { return '""' }
    if ($Argument -notmatch '[\s"]') { return $Argument }
    $builder = New-Object Text.StringBuilder
    [void]$builder.Append('"')
    $slashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $slashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append((('\' * ($slashes * 2 + 1)) -join ''))
            [void]$builder.Append('"')
            $slashes = 0
            continue
        }
        if ($slashes -gt 0) { [void]$builder.Append((('\' * $slashes) -join '')); $slashes = 0 }
        [void]$builder.Append($character)
    }
    if ($slashes -gt 0) { [void]$builder.Append((('\' * ($slashes * 2)) -join '')) }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Format-LauncherCommand {
    param([Parameter(Mandatory = $true)]$Command)
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $Command.Environment.GetEnumerator()) {
        $displayValue = ([string]$entry.Value).Replace("'", "''")
        $lines.Add(("ENV {0}='{1}'" -f $entry.Key, $displayValue))
    }
    $parts = New-Object System.Collections.Generic.List[string]
    $parts.Add((Quote-WindowsArgument $Command.FileName))
    foreach ($argument in $Command.Arguments) { $parts.Add((Quote-WindowsArgument $argument)) }
    $lines.Add(("COMMAND " + ($parts -join ' ')))
    return $lines -join [Environment]::NewLine
}

function New-LauncherProcessStartInfo {
    param([Parameter(Mandatory = $true)]$Command)
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Command.FileName
    $info.Arguments = (($Command.Arguments | ForEach-Object { Quote-WindowsArgument $_ }) -join ' ')
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.WorkingDirectory = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Command.FileName))
    foreach ($entry in $Command.Environment.GetEnumerator()) { $info.EnvironmentVariables[$entry.Key] = $entry.Value }
    return $info
}

function Save-LauncherProfile {
    param([Parameter(Mandatory = $true)]$Profile, [Parameter(Mandatory = $true)][string]$Path)
    $parent = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Path))
    if (-not [IO.Directory]::Exists($parent)) { [IO.Directory]::CreateDirectory($parent) | Out-Null }
    $temporary = "$Path.tmp.$PID"
    (Merge-LauncherProfile $Profile) | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Load-LauncherProfile {
    param([Parameter(Mandatory = $true)][string]$Path)
    $loaded = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    $profile = Merge-LauncherProfile $loaded
    foreach ($name in @("NoWarmup", "EnableNumaEp", "EnableHierBarrier", "EnableGateUpParallel", "EnableRemoteEp", "RemoteMaxEffort", "RemotePp", "RemoteRdma", "RemoteWeightOnMaster", "RemoteParallelIo")) {
        if ($profile[$name] -isnot [bool]) { throw "$name must be a JSON boolean" }
    }
    return $profile
}

function Invoke-LauncherCoreSelfTest {
    $profile = New-LauncherProfile
    $profile.ModelPath = 'C:\Models With Space\model "safe".gguf'
    $profile.ServerPath = 'C:\Program Files\PeoplesLLM\llama-server.exe'
    $profile.DSparkPath = 'D:\draft models\dspark.gguf'
    $profile.EnableRemoteEp = $true
    $profile.RemoteEndpoints = '127.0.0.1:29200,127.0.0.1:29201'
    $command = ConvertTo-LauncherCommand $profile
    if ($command.Arguments[0] -ne '--model' -or $command.Arguments[1] -ne $profile.ModelPath) { throw "argument tokenization failed" }
    if ($command.Environment['GGML_REMOTE_EP_SCHED_KLOCAL'] -ne '0') { throw "environment mapping failed" }
    if ((Quote-WindowsArgument 'C:\a b\') -ne '"C:\a b\\"') { throw "trailing slash quoting failed" }
    if ((Quote-WindowsArgument 'a"b') -ne '"a\"b"') { throw "embedded quote quoting failed" }
    $preview = Format-LauncherCommand $command
    if ($preview -notmatch '--spec-type draft-dspark' -or $preview -notmatch "GGML_REMOTE_EP='1'") { throw "preview failed" }
    $temp = [IO.Path]::Combine([IO.Path]::GetTempPath(), "peoplesllm-profile-$PID.json")
    try {
        Save-LauncherProfile $profile $temp
        $roundtrip = Load-LauncherProfile $temp
        if ($roundtrip.ModelPath -ne $profile.ModelPath -or -not $roundtrip.EnableRemoteEp) { throw "profile roundtrip failed" }
    } finally {
        if ([IO.File]::Exists($temp)) { Remove-Item -LiteralPath $temp -Force }
    }
    return "PeoplesLLM launcher core selftest: PASS"
}

Export-ModuleMember -Function New-LauncherProfile, Merge-LauncherProfile, Test-LauncherProfile, ConvertTo-LauncherCommand, Quote-WindowsArgument, Format-LauncherCommand, New-LauncherProcessStartInfo, Save-LauncherProfile, Load-LauncherProfile, Invoke-LauncherCoreSelfTest
