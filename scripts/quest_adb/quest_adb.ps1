# Quest wireless ADB helper (Windows PowerShell)
# Edit CONFIG below once, then:
#   .\quest_adb.ps1 usb-wifi          # USB plugged: enable tcpip 5555
#   .\quest_adb.ps1 connect           # adb connect QUEST_IP:5555
#   .\quest_adb.ps1 devices
#   .\quest_adb.ps1 calib | practice | main
#   .\quest_adb.ps1 quit-calib | quit-practice | quit-main | quit-all
#   .\quest_adb.ps1 switch calib|practice|main   # force-stop all study pkgs, then launch

# ---------- CONFIG (edit these) ----------
$Adb = "adb"
$QuestIp = "192.168.x.x"   # Quest Wi-Fi IP
$AdbPort = 5555

$PkgCalib = "org.MixedRealityToolkit.MRTK3Sample"
$PkgPractice = "com.PracticeMG.MRstressPRACTICE"
$PkgMain = "com.PracticeMG.MRstress"
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
  calib | practice | main

Quit:
  quit-calib | quit-practice | quit-main | quit-all

Switch (quit all three, then launch one):
  switch calib|practice|main
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
}

function Switch-To([string]$Target) {
    Quit-All
    Start-Sleep -Milliseconds 400
    switch ($Target.ToLowerInvariant()) {
        "calib" { Launch-Pkg $PkgCalib }
        "practice" { Launch-Pkg $PkgPractice }
        "main" { Launch-Pkg $PkgMain }
        default { throw "switch target must be calib|practice|main (got '$Target')" }
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

    "quit-calib" { Quit-Pkg $PkgCalib }
    "quit-practice" { Quit-Pkg $PkgPractice }
    "quit-main" { Quit-Pkg $PkgMain }
    "quit-all" { Quit-All }

    "switch" {
        if (-not $arg1) { throw "usage: .\quest_adb.ps1 switch calib|practice|main" }
        Switch-To $arg1
    }

    default {
        Show-Usage
        throw "Unknown command: $cmd"
    }
}
