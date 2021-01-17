#!/usr/bin/pwsh

cd $PSScriptRoot

dir ./output/runs -File -Recurse -Exclude *.ps1 | ?{$_.Length -lt 32kb} |   %{echo "rm:    $_"; ri $_} # rm files by size
dir ./output/runs -Directory | ?{(dir $_.FullName | measure).Count -eq 0} | %{echo "rmdir: $_"; ri $_} # rm empty folders
