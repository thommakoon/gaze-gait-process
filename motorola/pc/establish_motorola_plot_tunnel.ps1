#Requires -Version 5.1
<#
.SYNOPSIS
  Reverse SSH tunnel: Termux (phone) TCP port -> this PC's plot listener.

.DESCRIPTION
  Run this on the PC *after* starting tcp_dual_imu_rpy_plot.py on -LocalPlotPort,
  *before* starting neon_imu_recorder_dual.py on the phone.

  Flow:
    1) PC:  python tcp_dual_imu_rpy_plot.py --listen 127.0.0.1 --port <LocalPlotPort>
    2) PC:  .\establish_motorola_plot_tunnel.ps1 -PhoneHost ... -PhoneUser ...
    3) Termux: python neon_imu_recorder_dual.py --forward-tcp 127.0.0.1:<RemoteForwardPort>

  Termux must run sshd (pkg install openssh; sshd). PC needs OpenSSH client (ssh).

  The tunnel is: connections to 127.0.0.1:RemoteForwardPort ON THE PHONE are
  forwarded to 127.0.0.1:LocalPlotPort on this PC.

.EXAMPLE
  .\establish_motorola_plot_tunnel.ps1 -PhoneHost 192.168.1.42 -PhoneUser u0_a252
#>

param(
    [Parameter(Mandatory = $true)]
    [string] $PhoneHost,

    [Parameter(Mandatory = $true)]
    [string] $PhoneUser,

    [int] $SshPort = 22,
    [int] $RemoteForwardPort = 9001,
    [int] $LocalPlotPort = 9002,
    [string[]] $SshExtraArgs = @()
)

$remoteSpec = "${RemoteForwardPort}:127.0.0.1:${LocalPlotPort}"
Write-Host "Opening reverse tunnel on phone: 127.0.0.1:${RemoteForwardPort} -> PC 127.0.0.1:${LocalPlotPort}"
Write-Host "On Termux use: --forward-tcp 127.0.0.1:${RemoteForwardPort}"
Write-Host "SSH to phone: port ${SshPort} (Termux often uses 8022 — pass -SshPort 8022)"
Write-Host "Press Ctrl+C to close the tunnel."
Write-Host ""

$sshArgs = @(
    "-N",
    "-p", "$SshPort",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
    "-R", $remoteSpec,
    "${PhoneUser}@${PhoneHost}"
) + $SshExtraArgs

& ssh @sshArgs
