#!/usr/bin/pwsh

cd $PSScriptRoot

dir ./runs -File -Recurse -Exclude *.ps1 | ?{$_.Length -lt 10kb} |   %{echo "rm:    $_"; ri $_} # files that size < 10KB
dir ./runs -Directory | ?{(dir $_.FullName | measure).Count -eq 0} | %{echo "rmdir: $_"; ri $_} # empty folders
