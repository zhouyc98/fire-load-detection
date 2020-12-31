dir ./runs -File -Recurse | ?{$_.Length -lt 1000} | %{write "rm: $_"; ri $_.FullName} # delete files that size < 1KB
dir ./runs -Directory | ?{(dir $_.FullName | measure).Count -eq 0} | %{write "rmdir: $_"; ri $_.FullName} # delete empty folders
