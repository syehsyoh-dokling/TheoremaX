<#
.SYNOPSIS
    Skrip PowerShell Pendaftaran & Pemaketan Submission Track 1 Agenthon 2026 (CodaBench)
.DESCRIPTION
    Skrip ini memaketkan descriptor submission_t1_final_know95.json menjadi arsip terotentikasi
    resmi track1-final-know95-20261026.zip menggunakan alat resmi qfbench2 di WSL.
#>

[CmdletBinding()]
param(
    [string]$TargetFolder = "C:\Users\Saifuddin\Documents\Agenthon-2026\submissions\2026-10-26",
    [string]$Descriptor = "submission_t1_final_know95.json",
    [string]$ZipName = "track1-final-know95-20261026.zip",
    [int]$TeamNumber = 210
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   SKRIP PENDAFTARAN & PEMAKETAN SUBMISSION TRACK 1 (FINAL know95) " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

if (-not (Test-Path $TargetFolder)) {
    New-Item -ItemType Directory -Path $TargetFolder -Force | Out-Null
}

$drive = $TargetFolder.Substring(0, 1).ToLower()
$wslDir = "/mnt/$drive" + $TargetFolder.Substring(2).Replace("\", "/")
$descPath = Join-Path $TargetFolder $Descriptor
$outZipPath = Join-Path $TargetFolder $ZipName

if (-not (Test-Path $descPath)) {
    Write-Host "[ERROR] Descriptor tidak ditemukan: $descPath" -ForegroundColor Red
    exit 1
}

$descContent = Get-Content $descPath -Raw | ConvertFrom-Json
Write-Host "`n[1/3] Memeriksa Metadata Descriptor:" -ForegroundColor Yellow
Write-Host "  - Track          : $($descContent.track)"
Write-Host "  - Phase          : $($descContent.phase)"
Write-Host "  - Competition ID : $($descContent.competition_id)"
Write-Host "  - Image Registry : $($descContent.image.registry)"
Write-Host "  - Repository     : $($descContent.image.repository)"
Write-Host "  - Digest SHA256  : $($descContent.image.digest)"

Write-Host "`n[2/3] Memaketkan Signature Kriptografi Tim $TeamNumber via qfbench2..." -ForegroundColor Yellow

$wslCmd = "cd '$wslDir' && if [ -f ~/.team_key ]; then ~/venvs/agenthon/bin/qfbench2 submission pack --team-number $TeamNumber --team-key-file ~/.team_key --descriptor $Descriptor --out $ZipName --force; else ~/venvs/agenthon/bin/qfbench2 submission pack --team-number $TeamNumber --descriptor $Descriptor --out $ZipName --force; fi"

wsl -e bash -lc $wslCmd

if (Test-Path $outZipPath) {
    Write-Host "`n[3/3] Verifikasi Hasil Berkas ZIP:" -ForegroundColor Green
    $zipItem = Get-Item $outZipPath
    $hash = Get-FileHash -Path $outZipPath -Algorithm SHA256
    
    $downloadsPath = "C:\Users\Saifuddin\Downloads\$ZipName"
    Copy-Item $outZipPath -Destination $downloadsPath -Force
    
    Write-Host "  - Berkas Utama : $outZipPath ($($zipItem.Length) bytes)" -ForegroundColor Green
    Write-Host "  - Salinan Unduh: $downloadsPath" -ForegroundColor Green
    Write-Host "  - Checksum     : $($hash.Hash)" -ForegroundColor Green
    
    Write-Host "`n=================================================================" -ForegroundColor Cyan
    Write-Host "   STATUS: PAKET BERHASIL DIBUAT DAN SIAP DI-SUBMIT!            " -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host "`nInstruksi Submit ke CodaBench:" -ForegroundColor Yellow
    Write-Host "1. Buka browser: https://www.codabench.org/competitions/ (cari kompetisi Agenthon 2026 Track 1 FINAL; Dev-nya 17765)"
    Write-Host "2. Masuk ke tab 'My Submissions'"
    Write-Host "3. Klik tombol upload dan pilih file:"
    Write-Host "   $downloadsPath" -ForegroundColor White
} else {
    Write-Host "`n[ERROR] Gagal membuat file zip!" -ForegroundColor Red
    exit 1
}
