#!/usr/bin/pwsh

cd $PSScriptRoot

dir ./runs -File -Recurse -Exclude *.ps1 | ?{$_.Length -lt 32kb} |   %{echo "rm:    $_"; ri $_} # rm files by size
dir ./runs -Directory | ?{(dir $_.FullName | measure).Count -eq 0} | %{echo "rmdir: $_"; ri $_} # rm empty folders
