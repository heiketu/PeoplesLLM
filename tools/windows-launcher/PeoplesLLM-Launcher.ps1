[CmdletBinding()]
param(
    [string]$Profile = "",
    [switch]$DryRun,
    [switch]$SelfTest
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$modulePath = Join-Path $PSScriptRoot "PeoplesLLM.Launcher.psm1"
Import-Module $modulePath -Force

if ($SelfTest) {
    Write-Output (Invoke-LauncherCoreSelfTest)
    exit 0
}

$initialProfile = if ([string]::IsNullOrWhiteSpace($Profile)) { New-LauncherProfile } else { Load-LauncherProfile $Profile }
if ($DryRun) {
    $errors = @(Test-LauncherProfile $initialProfile)
    if ($errors.Count -gt 0) {
        foreach ($message in $errors) { [Console]::Error.WriteLine("error: $message") }
        exit 2
    }
    $command = ConvertTo-LauncherCommand $initialProfile
    Write-Output (Format-LauncherCommand $command)
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Net.Http
[Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object Windows.Forms.Form
$form.Text = "PeoplesLLM Server Launcher"
$form.StartPosition = "CenterScreen"
$form.Size = New-Object Drawing.Size(1280, 900)
$form.MinimumSize = New-Object Drawing.Size(1100, 760)

$tabs = New-Object Windows.Forms.TabControl
$tabs.Dock = "Fill"
$configTab = New-Object Windows.Forms.TabPage
$configTab.Text = "Configuration"
$previewTab = New-Object Windows.Forms.TabPage
$previewTab.Text = "Command preview"
$runtimeTab = New-Object Windows.Forms.TabPage
$runtimeTab.Text = "Logs and chat"
[void]$tabs.TabPages.Add($configTab)
[void]$tabs.TabPages.Add($previewTab)
[void]$tabs.TabPages.Add($runtimeTab)

$topPanel = New-Object Windows.Forms.Panel
$topPanel.Dock = "Top"
$topPanel.Height = 48
$bottomPanel = New-Object Windows.Forms.Panel
$bottomPanel.Dock = "Bottom"
$bottomPanel.Height = 48
$form.Controls.Add($tabs)
$form.Controls.Add($bottomPanel)
$form.Controls.Add($topPanel)

function New-Button {
    param($Parent, [string]$Text, [int]$X, [int]$Y, [int]$Width = 105)
    $button = New-Object Windows.Forms.Button
    $button.Text = $Text
    $button.Location = New-Object Drawing.Point($X, $Y)
    $button.Size = New-Object Drawing.Size($Width, 30)
    $Parent.Controls.Add($button)
    return $button
}

function New-TextField {
    param($Parent, [string]$Name, [string]$Label, [int]$X, [int]$Y, [int]$Width, [switch]$Multiline)
    $caption = New-Object Windows.Forms.Label
    $caption.Text = $Label
    $caption.Location = New-Object Drawing.Point($X, $Y)
    $caption.Size = New-Object Drawing.Size($Width, 18)
    $Parent.Controls.Add($caption)
    $field = New-Object Windows.Forms.TextBox
    $field.Name = $Name
    $field.Location = New-Object Drawing.Point($X, ($Y + 19))
    $height = if ($Multiline) { 82 } else { 24 }
    $field.Size = New-Object Drawing.Size($Width, $height)
    $field.Multiline = [bool]$Multiline
    if ($Multiline) { $field.ScrollBars = "Vertical"; $field.AcceptsReturn = $true }
    $Parent.Controls.Add($field)
    return $field
}

function New-ComboField {
    param($Parent, [string]$Name, [string]$Label, [int]$X, [int]$Y, [int]$Width, [string[]]$Items)
    $caption = New-Object Windows.Forms.Label
    $caption.Text = $Label
    $caption.Location = New-Object Drawing.Point($X, $Y)
    $caption.Size = New-Object Drawing.Size($Width, 18)
    $Parent.Controls.Add($caption)
    $field = New-Object Windows.Forms.ComboBox
    $field.Name = $Name
    $field.DropDownStyle = "DropDownList"
    $field.Location = New-Object Drawing.Point($X, ($Y + 19))
    $field.Size = New-Object Drawing.Size($Width, 24)
    [void]$field.Items.AddRange($Items)
    $Parent.Controls.Add($field)
    return $field
}

function New-CheckField {
    param($Parent, [string]$Name, [string]$Label, [int]$X, [int]$Y, [int]$Width = 180)
    $field = New-Object Windows.Forms.CheckBox
    $field.Name = $Name
    $field.Text = $Label
    $field.Location = New-Object Drawing.Point($X, $Y)
    $field.Size = New-Object Drawing.Size($Width, 24)
    $Parent.Controls.Add($field)
    return $field
}

function New-SectionLabel {
    param($Parent, [string]$Text, [int]$Y)
    $label = New-Object Windows.Forms.Label
    $label.Text = $Text
    $label.Font = New-Object Drawing.Font($form.Font, [Drawing.FontStyle]::Bold)
    $label.Location = New-Object Drawing.Point(12, $Y)
    $label.Size = New-Object Drawing.Size(1160, 22)
    $Parent.Controls.Add($label)
}

$configPanel = New-Object Windows.Forms.Panel
$configPanel.Dock = "Fill"
$configPanel.AutoScroll = $true
$configTab.Controls.Add($configPanel)
$controls = @{}

New-SectionLabel $configPanel "Files and network" 10
$controls.ServerPath = New-TextField $configPanel "ServerPath" "llama-server.exe" 12 36 1030
$browseServer = New-Button $configPanel "Browse..." 1055 53 100
$controls.ModelPath = New-TextField $configPanel "ModelPath" "Main model GGUF" 12 82 1030
$browseModel = New-Button $configPanel "Browse..." 1055 99 100
$controls.DSparkPath = New-TextField $configPanel "DSparkPath" "DSpark GGUF (blank disables speculative decoding)" 12 128 1030
$browseDSpark = New-Button $configPanel "Browse..." 1055 145 100
$controls.ModelAlias = New-TextField $configPanel "ModelAlias" "API model alias" 12 176 230
$controls.Host = New-TextField $configPanel "Host" "Bind host" 255 176 230
$controls.Port = New-TextField $configPanel "Port" "Port" 498 176 130
$controls.ContextSize = New-TextField $configPanel "ContextSize" "Context per server" 641 176 170
$controls.Slots = New-TextField $configPanel "Slots" "Slots" 824 176 100

New-SectionLabel $configPanel "CPU, GPU and KV" 228
$controls.Threads = New-TextField $configPanel "Threads" "Threads (0=auto)" 12 254 140
$controls.ThreadsBatch = New-TextField $configPanel "ThreadsBatch" "Batch threads (0=auto)" 165 254 170
$controls.BatchSize = New-TextField $configPanel "BatchSize" "Batch size" 348 254 130
$controls.UBatchSize = New-TextField $configPanel "UBatchSize" "Ubatch size" 491 254 130
$controls.Device = New-TextField $configPanel "Device" "Device list (none/CUDA0,CUDA1)" 634 254 250
$controls.GpuLayers = New-TextField $configPanel "GpuLayers" "GPU layers (0/auto/all)" 897 254 170
$controls.CpuMoeLayers = New-TextField $configPanel "CpuMoeLayers" "CPU MoE layers" 1080 254 110
$controls.SplitMode = New-ComboField $configPanel "SplitMode" "GPU split mode" 12 303 140 @("none", "layer", "row", "tensor")
$controls.TensorSplit = New-TextField $configPanel "TensorSplit" "Tensor split (for example 1,1)" 165 303 190
$controls.MainGpu = New-TextField $configPanel "MainGpu" "Main GPU" 368 303 100
$controls.Numa = New-ComboField $configPanel "Numa" "NUMA" 481 303 150 @("none", "distribute", "isolate", "numactl", "mirror")
$controls.NumaMirror = New-ComboField $configPanel "NumaMirror" "Mirror components" 644 303 160 @("all", "weights", "kv", "none")
$controls.LoadMode = New-ComboField $configPanel "LoadMode" "Load mode" 817 303 150 @("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")
$controls.CacheTypeK = New-ComboField $configPanel "CacheTypeK" "KV K type" 980 303 100 @("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
$controls.CacheTypeV = New-ComboField $configPanel "CacheTypeV" "KV V type" 1093 303 100 @("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
$controls.FlashAttention = New-ComboField $configPanel "FlashAttention" "Flash attention" 12 352 140 @("auto", "on", "off")
$controls.NoWarmup = New-CheckField $configPanel "NoWarmup" "Disable warmup" 165 371 150
$controls.EnableNumaEp = New-CheckField $configPanel "EnableNumaEp" "Enable CPU NUMA EP" 328 371 180
$controls.EnableHierBarrier = New-CheckField $configPanel "EnableHierBarrier" "Hierarchical barrier" 521 371 180
$controls.EnableGateUpParallel = New-CheckField $configPanel "EnableGateUpParallel" "Parallel gate/up" 714 371 160

New-SectionLabel $configPanel "DSpark" 410
$controls.DSparkNMax = New-TextField $configPanel "DSparkNMax" "Draft n-max" 12 436 140
$controls.DSparkPMin = New-TextField $configPanel "DSparkPMin" "Draft p-min" 165 436 140
$controls.DraftDevice = New-TextField $configPanel "DraftDevice" "Draft device" 318 436 220
$controls.DraftGpuLayers = New-TextField $configPanel "DraftGpuLayers" "Draft GPU layers" 551 436 170

New-SectionLabel $configPanel "Scheduled remote EP (workers must be started separately)" 486
$controls.EnableRemoteEp = New-CheckField $configPanel "EnableRemoteEp" "Enable remote EP" 12 512 160
$controls.RemoteMaxEffort = New-CheckField $configPanel "RemoteMaxEffort" "Max effort / replicas" 185 512 180
$controls.RemotePp = New-CheckField $configPanel "RemotePp" "Enable PP" 378 512 120
$controls.RemoteRdma = New-CheckField $configPanel "RemoteRdma" "RDMA" 511 512 100
$controls.RemoteWeightOnMaster = New-CheckField $configPanel "RemoteWeightOnMaster" "Weight on master" 624 512 170
$controls.RemoteParallelIo = New-CheckField $configPanel "RemoteParallelIo" "Parallel endpoint I/O" 807 512 190
$controls.RemoteEndpoints = New-TextField $configPanel "RemoteEndpoints" "Endpoints (host:port,host:port)" 12 544 410
$controls.RemoteLayers = New-TextField $configPanel "RemoteLayers" "Layers" 435 544 140
$controls.RemoteKLocal = New-TextField $configPanel "RemoteKLocal" "K local (0=pure EP)" 588 544 160
$controls.RemoteReconnectTimeoutMs = New-TextField $configPanel "RemoteReconnectTimeoutMs" "Reconnect timeout ms" 761 544 190

New-SectionLabel $configPanel "Advanced" 596
$controls.ExtraArguments = New-TextField $configPanel "ExtraArguments" "Extra arguments: one complete argv token per line (no shell parsing)" 12 622 570 -Multiline
$controls.ExtraEnvironment = New-TextField $configPanel "ExtraEnvironment" "Extra environment: KEY=VALUE, one per line" 595 622 570 -Multiline
$configPanel.AutoScrollMinSize = New-Object Drawing.Size(1180, 740)

$previewBox = New-Object Windows.Forms.TextBox
$previewBox.Dock = "Fill"
$previewBox.Multiline = $true
$previewBox.ScrollBars = "Both"
$previewBox.ReadOnly = $true
$previewBox.WordWrap = $false
$previewBox.Font = New-Object Drawing.Font("Consolas", 10)
$previewRefresh = New-Button $previewTab "Refresh preview" 8 8 130
$previewCopy = New-Button $previewTab "Copy" 150 8 90
$previewPanel = New-Object Windows.Forms.Panel
$previewPanel.Dock = "Top"
$previewPanel.Height = 46
$previewTab.Controls.Add($previewBox)
$previewTab.Controls.Add($previewPanel)
$previewPanel.Controls.Add($previewRefresh)
$previewPanel.Controls.Add($previewCopy)

$runtimeSplit = New-Object Windows.Forms.SplitContainer
$runtimeSplit.Dock = "Fill"
$runtimeSplit.Orientation = "Horizontal"
$runtimeSplit.SplitterDistance = 390
$runtimeTab.Controls.Add($runtimeSplit)
$logBox = New-Object Windows.Forms.TextBox
$logBox.Dock = "Fill"
$logBox.Multiline = $true
$logBox.ReadOnly = $true
$logBox.ScrollBars = "Both"
$logBox.WordWrap = $false
$logBox.Font = New-Object Drawing.Font("Consolas", 9)
$runtimeSplit.Panel1.Controls.Add($logBox)
$chatOutput = New-Object Windows.Forms.RichTextBox
$chatOutput.Dock = "Fill"
$chatOutput.ReadOnly = $true
$chatPanel = New-Object Windows.Forms.Panel
$chatPanel.Dock = "Bottom"
$chatPanel.Height = 102
$chatInput = New-Object Windows.Forms.TextBox
$chatInput.Multiline = $true
$chatInput.ScrollBars = "Vertical"
$chatInput.Location = New-Object Drawing.Point(8, 8)
$chatInput.Size = New-Object Drawing.Size(960, 84)
$chatSend = New-Button $chatPanel "Send" 980 8 100
$chatReset = New-Button $chatPanel "Reset chat" 980 48 100
$chatPanel.Controls.Add($chatInput)
$runtimeSplit.Panel2.Controls.Add($chatOutput)
$runtimeSplit.Panel2.Controls.Add($chatPanel)

$loadProfile = New-Button $topPanel "Load profile" 8 8 110
$saveProfile = New-Button $topPanel "Save profile" 126 8 110
$resetProfile = New-Button $topPanel "Defaults" 244 8 100
$openReadme = New-Button $topPanel "README" 352 8 100
$startButton = New-Button $bottomPanel "Start server" 8 8 120
$stopButton = New-Button $bottomPanel "Stop server" 136 8 120
$stopButton.Enabled = $false
$healthLabel = New-Object Windows.Forms.Label
$healthLabel.Text = "Stopped"
$healthLabel.Location = New-Object Drawing.Point(275, 14)
$healthLabel.Size = New-Object Drawing.Size(850, 24)
$bottomPanel.Controls.Add($healthLabel)

function Set-ComboValue {
    param($Combo, [string]$Value)
    $index = $Combo.Items.IndexOf($Value)
    if ($index -lt 0) { $index = 0 }
    $Combo.SelectedIndex = $index
}

function Set-ControlsFromProfile {
    param($InputProfile)
    $profileValue = Merge-LauncherProfile $InputProfile
    foreach ($name in @("ServerPath", "ModelPath", "ModelAlias", "DSparkPath", "Host", "Port", "ContextSize", "Slots", "Threads", "ThreadsBatch", "BatchSize", "UBatchSize", "Device", "GpuLayers", "CpuMoeLayers", "TensorSplit", "MainGpu", "NumaMirror", "CacheTypeK", "CacheTypeV", "DSparkNMax", "DSparkPMin", "DraftDevice", "DraftGpuLayers", "RemoteEndpoints", "RemoteLayers", "RemoteKLocal", "RemoteReconnectTimeoutMs", "ExtraEnvironment")) {
        $controls[$name].Text = [string]$profileValue[$name]
    }
    $controls.ExtraArguments.Text = (@($profileValue.ExtraArguments) -join [Environment]::NewLine)
    foreach ($name in @("SplitMode", "Numa", "LoadMode", "FlashAttention")) { Set-ComboValue $controls[$name] ([string]$profileValue[$name]) }
    foreach ($name in @("NoWarmup", "EnableNumaEp", "EnableHierBarrier", "EnableGateUpParallel", "EnableRemoteEp", "RemoteMaxEffort", "RemotePp", "RemoteRdma", "RemoteWeightOnMaster", "RemoteParallelIo")) {
        $controls[$name].Checked = [bool]$profileValue[$name]
    }
}

function Get-ProfileFromControls {
    $profileValue = New-LauncherProfile
    foreach ($name in @("ServerPath", "ModelPath", "ModelAlias", "DSparkPath", "Host", "Device", "GpuLayers", "TensorSplit", "NumaMirror", "CacheTypeK", "CacheTypeV", "DraftDevice", "DraftGpuLayers", "RemoteEndpoints", "RemoteLayers", "ExtraEnvironment")) {
        $profileValue[$name] = $controls[$name].Text.Trim()
    }
    foreach ($name in @("Port", "ContextSize", "Slots", "Threads", "ThreadsBatch", "BatchSize", "UBatchSize", "CpuMoeLayers", "MainGpu", "DSparkNMax", "RemoteKLocal", "RemoteReconnectTimeoutMs")) {
        $profileValue[$name] = $controls[$name].Text.Trim()
    }
    $profileValue.DSparkPMin = $controls.DSparkPMin.Text.Trim()
    foreach ($name in @("SplitMode", "Numa", "LoadMode", "FlashAttention")) { $profileValue[$name] = [string]$controls[$name].SelectedItem }
    foreach ($name in @("NoWarmup", "EnableNumaEp", "EnableHierBarrier", "EnableGateUpParallel", "EnableRemoteEp", "RemoteMaxEffort", "RemotePp", "RemoteRdma", "RemoteWeightOnMaster", "RemoteParallelIo")) {
        $profileValue[$name] = $controls[$name].Checked
    }
    $profileValue.ExtraArguments = @($controls.ExtraArguments.Lines | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    return $profileValue
}

function Show-ValidationErrors {
    param([string[]]$Errors)
    if ($null -ne $Errors -and $Errors.Count -gt 0) {
        [Windows.Forms.MessageBox]::Show(($Errors -join [Environment]::NewLine), "Invalid configuration", "OK", "Error") | Out-Null
        return $true
    }
    return $false
}

function Update-CommandPreview {
    try {
        $profileValue = Get-ProfileFromControls
        $errors = @(Test-LauncherProfile $profileValue)
        $text = ""
        if ($errors.Count -gt 0) { $text = "VALIDATION WARNINGS:`r`n" + ($errors -join "`r`n") + "`r`n`r`n" }
        $previewBox.Text = $text + (Format-LauncherCommand (ConvertTo-LauncherCommand $profileValue))
    } catch {
        $previewBox.Text = "ERROR: " + $_.Exception.Message
    }
}

function Select-FileInto {
    param($TextBox, [string]$Filter)
    $dialog = New-Object Windows.Forms.OpenFileDialog
    $dialog.Filter = $Filter
    $dialog.CheckFileExists = $true
    if (-not [string]::IsNullOrWhiteSpace($TextBox.Text)) { $dialog.FileName = $TextBox.Text }
    if ($dialog.ShowDialog() -eq "OK") { $TextBox.Text = $dialog.FileName }
    $dialog.Dispose()
}

function Append-Log {
    param([string]$Line)
    if ([string]::IsNullOrEmpty($Line)) { return }
    $logBox.AppendText($Line + [Environment]::NewLine)
    $logBox.SelectionStart = $logBox.TextLength
    $logBox.ScrollToCaret()
}

$script:ServerProcess = $null
$script:LogQueue = New-Object 'Collections.Concurrent.ConcurrentQueue[string]'
$script:HealthTask = $null
$script:ChatTask = $null
$script:ChatContent = $null
$script:PendingUserMessage = $null
$script:ChatMessages = New-Object Collections.ArrayList
$healthClient = New-Object Net.Http.HttpClient
$healthClient.Timeout = [TimeSpan]::FromSeconds(2)
$chatClient = New-Object Net.Http.HttpClient
$chatClient.Timeout = [TimeSpan]::FromHours(1)

function Get-ApiBase {
    $hostValue = $controls.Host.Text.Trim()
    if ($hostValue -eq "0.0.0.0" -or $hostValue -eq "::" -or $hostValue -eq "[::]") { $hostValue = "127.0.0.1" }
    if ($hostValue.Contains(":") -and -not $hostValue.StartsWith("[")) { $hostValue = "[$hostValue]" }
    return "http://${hostValue}:$($controls.Port.Text.Trim())"
}

function Stop-ServerProcess {
    if ($null -eq $script:ServerProcess) { return }
    try {
        if (-not $script:ServerProcess.HasExited) {
            Append-Log "[launcher] stopping server PID $($script:ServerProcess.Id)"
            $script:ServerProcess.Kill()
            [void]$script:ServerProcess.WaitForExit(5000)
        }
    } catch {
        Append-Log ("[launcher] stop error: " + $_.Exception.Message)
    } finally {
        try { $script:ServerProcess.Dispose() } catch {}
        $script:ServerProcess = $null
        $startButton.Enabled = $true
        $stopButton.Enabled = $false
        $healthLabel.Text = "Stopped"
    }
}

function Start-ConfiguredServer {
    $profileValue = Get-ProfileFromControls
    $errors = @(Test-LauncherProfile $profileValue -RequireFiles)
    if (Show-ValidationErrors $errors) { return }
    Stop-ServerProcess
    try {
        $command = ConvertTo-LauncherCommand $profileValue
        $previewBox.Text = Format-LauncherCommand $command
        $process = New-Object Diagnostics.Process
        $process.StartInfo = New-LauncherProcessStartInfo $command
        $process.EnableRaisingEvents = $true
        $process.add_OutputDataReceived({ param($sender, $eventArgs) if ($null -ne $eventArgs.Data) { $script:LogQueue.Enqueue($eventArgs.Data) } })
        $process.add_ErrorDataReceived({ param($sender, $eventArgs) if ($null -ne $eventArgs.Data) { $script:LogQueue.Enqueue($eventArgs.Data) } })
        if (-not $process.Start()) { throw "Process.Start returned false" }
        $script:ServerProcess = $process
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()
        $startButton.Enabled = $false
        $stopButton.Enabled = $true
        $healthLabel.Text = "Starting, PID $($process.Id)"
        Append-Log "[launcher] started PID $($process.Id)"
        $tabs.SelectedTab = $runtimeTab
    } catch {
        Append-Log ("[launcher] start failed: " + $_.Exception.Message)
        [Windows.Forms.MessageBox]::Show($_.Exception.Message, "Start failed", "OK", "Error") | Out-Null
        Stop-ServerProcess
    }
}

$uiTimer = New-Object Windows.Forms.Timer
$uiTimer.Interval = 250
$uiTimer.Add_Tick({
    $line = $null
    while ($script:LogQueue.TryDequeue([ref]$line)) { Append-Log $line; $line = $null }
    if ($null -ne $script:ServerProcess -and $script:ServerProcess.HasExited) {
        Append-Log "[launcher] server exited with code $($script:ServerProcess.ExitCode)"
        Stop-ServerProcess
    }
    if ($null -ne $script:ChatTask -and $script:ChatTask.IsCompleted) {
        $response = $null
        try {
            if ($script:ChatTask.IsFaulted) { throw ($script:ChatTask.Exception.GetBaseException()) }
            $response = $script:ChatTask.Result
            $body = $response.Content.ReadAsStringAsync().Result
            if (-not $response.IsSuccessStatusCode) { throw "HTTP $([int]$response.StatusCode): $body" }
            $decoded = $body | ConvertFrom-Json
            $assistantText = [string]$decoded.choices[0].message.content
            [void]$script:ChatMessages.Add([pscustomobject]@{ role = "user"; content = $script:PendingUserMessage })
            [void]$script:ChatMessages.Add([pscustomobject]@{ role = "assistant"; content = $assistantText })
            $chatOutput.AppendText("You:`r`n$($script:PendingUserMessage)`r`n`r`nAssistant:`r`n$assistantText`r`n`r`n")
        } catch {
            $chatOutput.AppendText("[chat error] $($_.Exception.Message)`r`n`r`n")
        } finally {
            if ($null -ne $response) { $response.Dispose() }
            if ($null -ne $script:ChatContent) { $script:ChatContent.Dispose() }
            $script:ChatTask = $null
            $script:ChatContent = $null
            $script:PendingUserMessage = $null
            $chatSend.Enabled = $true
        }
    }
})
$uiTimer.Start()

$healthTimer = New-Object Windows.Forms.Timer
$healthTimer.Interval = 1000
$healthTimer.Add_Tick({
    if ($null -eq $script:ServerProcess) { return }
    if ($null -ne $script:HealthTask) {
        if (-not $script:HealthTask.IsCompleted) { return }
        try {
            if ($script:HealthTask.IsFaulted) { throw ($script:HealthTask.Exception.GetBaseException()) }
            $result = $script:HealthTask.Result
            $healthLabel.Text = if ($result.IsSuccessStatusCode) { "Healthy - $([int]$result.StatusCode) - PID $($script:ServerProcess.Id)" } else { "Loading/unhealthy - HTTP $([int]$result.StatusCode)" }
            $result.Dispose()
        } catch {
            $healthLabel.Text = "Starting/unreachable - PID $($script:ServerProcess.Id)"
        } finally {
            $script:HealthTask = $null
        }
    } else {
        $script:HealthTask = $healthClient.GetAsync((Get-ApiBase) + "/health")
    }
})
$healthTimer.Start()

$browseServer.Add_Click({ Select-FileInto $controls.ServerPath "Executable (*.exe)|*.exe|All files (*.*)|*.*" })
$browseModel.Add_Click({ Select-FileInto $controls.ModelPath "GGUF model (*.gguf)|*.gguf|All files (*.*)|*.*" })
$browseDSpark.Add_Click({ Select-FileInto $controls.DSparkPath "GGUF model (*.gguf)|*.gguf|All files (*.*)|*.*" })
$previewRefresh.Add_Click({ Update-CommandPreview })
$previewCopy.Add_Click({ if (-not [string]::IsNullOrEmpty($previewBox.Text)) { [Windows.Forms.Clipboard]::SetText($previewBox.Text) } })
$startButton.Add_Click({ Start-ConfiguredServer })
$stopButton.Add_Click({ Stop-ServerProcess })
$resetProfile.Add_Click({ Set-ControlsFromProfile (New-LauncherProfile); Update-CommandPreview })
$openReadme.Add_Click({ Start-Process (Join-Path $PSScriptRoot "README.md") })

$loadProfile.Add_Click({
    $dialog = New-Object Windows.Forms.OpenFileDialog
    $dialog.Filter = "JSON profile (*.json)|*.json|All files (*.*)|*.*"
    if ($dialog.ShowDialog() -eq "OK") {
        try { Set-ControlsFromProfile (Load-LauncherProfile $dialog.FileName); Update-CommandPreview } catch { [Windows.Forms.MessageBox]::Show($_.Exception.Message, "Load failed", "OK", "Error") | Out-Null }
    }
    $dialog.Dispose()
})
$saveProfile.Add_Click({
    $dialog = New-Object Windows.Forms.SaveFileDialog
    $dialog.Filter = "JSON profile (*.json)|*.json|All files (*.*)|*.*"
    $dialog.DefaultExt = "json"
    if ($dialog.ShowDialog() -eq "OK") {
        try { Save-LauncherProfile (Get-ProfileFromControls) $dialog.FileName } catch { [Windows.Forms.MessageBox]::Show($_.Exception.Message, "Save failed", "OK", "Error") | Out-Null }
    }
    $dialog.Dispose()
})

$chatSend.Add_Click({
    if ($null -ne $script:ChatTask) { return }
    $message = $chatInput.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($message)) { return }
    try {
        $messages = @($script:ChatMessages) + @([pscustomobject]@{ role = "user"; content = $message })
        $payload = [ordered]@{ model = $controls.ModelAlias.Text.Trim(); messages = $messages; stream = $false }
        $json = $payload | ConvertTo-Json -Depth 8 -Compress
        $script:ChatContent = [Net.Http.StringContent]::new($json, [Text.Encoding]::UTF8, "application/json")
        $script:PendingUserMessage = $message
        $script:ChatTask = $chatClient.PostAsync((Get-ApiBase) + "/v1/chat/completions", $script:ChatContent)
        $chatInput.Clear()
        $chatSend.Enabled = $false
    } catch {
        $chatOutput.AppendText("[chat error] $($_.Exception.Message)`r`n")
    }
})
$chatReset.Add_Click({ $script:ChatMessages.Clear(); $chatOutput.Clear() })

$form.Add_FormClosing({
    $healthTimer.Stop()
    $uiTimer.Stop()
    Stop-ServerProcess
    if ($null -ne $script:ChatContent) { $script:ChatContent.Dispose() }
    $healthClient.Dispose()
    $chatClient.Dispose()
})

Set-ControlsFromProfile $initialProfile
Update-CommandPreview
[void]$form.ShowDialog()
