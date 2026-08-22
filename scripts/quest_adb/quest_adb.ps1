# Quest wireless ADB helper (Windows PowerShell)
# Edit CONFIG below once, then:
#   .\quest_adb.ps1 usb-wifi          # USB plugged: enable tcpip 5555
#   .\quest_adb.ps1 connect           # adb connect QUEST_IP:5555
#   .\quest_adb.ps1 devices
#   .\quest_adb.ps1 calib | practice | main | main-pro
#   .\quest_adb.ps1 quit-calib | quit-practice | quit-main | quit-main-pro | quit-all
#   .\quest_adb.ps1 switch calib|practice|main|main-pro   # force-stop all study pkgs, then launch

# ---------- CONFIG (edit these) ----------
$Adb = "adb"
$QuestIp = "192.168.x.x"   # Quest Wi-Fi IP
$AdbPort = 5555

$PkgCalib = "org.MixedRealityToolkit.MRTK3Sample"
$PkgPractice = "com.PracticeMG.MRstressPRACTICE"
$PkgMain = "com.PracticeMG.MRstress"           # Quest 3 + OpenEye
$PkgMainPro = "com.PracticeMG.MRstressPro"     # Quest Pro + OVR eye tracking
# ----------------------------------------

$ErrorActionPreference = "Stop"

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $Adb @Args
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed (exit $LASTEXITCODE): adb $($Args -join ' ')"
    }
}

function Show-Usage {
    @"
Usage: .\quest_adb.ps1 <command> [args]

Config: edit `$Adb and `$QuestIp at top of this script.

Setup (USB once per reboot if needed):
  usb-wifi     adb tcpip $AdbPort  (Quest on USB)
  connect      adb connect ${QuestIp}:$AdbPort
  devices      adb devices
  disconnect   adb disconnect ${QuestIp}:$AdbPort

Launch:
  calib | practice | main | main-pro

Quit:
  quit-calib | quit-practice | quit-main | quit-main-pro | quit-all

Switch (quit all study pkgs, then launch one):
  switch calib|practice|main|main-pro
"@
}

function Connect-Quest {
    Write-Host "Connecting ${QuestIp}:$AdbPort ..."
    Invoke-Adb connect "${QuestIp}:$AdbPort"
    Invoke-Adb devices
}

function Enable-UsbWifi {
    Write-Host "Enabling TCP mode on USB device (port $AdbPort)..."
    Invoke-Adb tcpip $AdbPort
    Write-Host "Unplug USB, then run: .\quest_adb.ps1 connect"
}

function Launch-Pkg([string]$Package) {
    Write-Host "Launching $Package ..."
    Invoke-Adb shell monkey -p $Package -c android.intent.category.LAUNCHER 1
}

function Quit-Pkg([string]$Package) {
    Write-Host "Force-stop $Package ..."
    Invoke-Adb shell am force-stop $Package
}

function Quit-All {
    Quit-Pkg $PkgCalib
    Quit-Pkg $PkgPractice
    Quit-Pkg $PkgMain
    Quit-Pkg $PkgMainPro
}

function Switch-To([string]$Target) {
    Quit-All
    Start-Sleep -Milliseconds 400
    switch ($Target.ToLowerInvariant()) {
        "calib" { Launch-Pkg $PkgCalib }
        "practice" { Launch-Pkg $PkgPractice }
        "main" { Launch-Pkg $PkgMain }
        "main-pro" { Launch-Pkg $PkgMainPro }
        default { throw "switch target must be calib|practice|main|main-pro (got '$Target')" }
    }
}

$cmd = if ($args.Count -ge 1) { $args[0].ToLowerInvariant() } else { "" }
$arg1 = if ($args.Count -ge 2) { $args[1] } else { "" }

switch ($cmd) {
    "" { Show-Usage }
    "help" { Show-Usage }
    "-h" { Show-Usage }
    "--help" { Show-Usage }

    "usb-wifi" { Enable-UsbWifi }
    "connect" { Connect-Quest }
    "devices" { Invoke-Adb devices }
    "disconnect" { Invoke-Adb disconnect "${QuestIp}:$AdbPort" }

    "calib" { Launch-Pkg $PkgCalib }
    "practice" { Launch-Pkg $PkgPractice }
    "main" { Launch-Pkg $PkgMain }
    "main-pro" { Launch-Pkg $PkgMainPro }

    "quit-calib" { Quit-Pkg $PkgCalib }
    "quit-practice" { Quit-Pkg $PkgPractice }
    "quit-main" { Quit-Pkg $PkgMain }
    "quit-main-pro" { Quit-Pkg $PkgMainPro }
    "quit-all" { Quit-All }

    "switch" {
        if (-not $arg1) { throw "usage: .\quest_adb.ps1 switch calib|practice|main|main-pro" }
        Switch-To $arg1
    }

    default {
        Show-Usage
        throw "Unknown command: $cmd"
    }
}
