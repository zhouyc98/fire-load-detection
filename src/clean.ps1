dir ./runs -File -Recurse -Exclude *.ps1 | ?{$_.Length -lt 1000} | ri # delete files that size < 1KB
dir ./runs -Directory | ?{(dir $_ | measure).Count -eq 0} | ri # delete empty folders
