cd $PSScriptRoot
echo 'Note: donot use this script when other GPUs are running'
dir ./runs -File -Recurse -Exclude *.ps1 | ?{$_.Length -lt 30000} | %{echo "rm: $_"; ri $_.FullName} # delete files that size < 1KB
dir ./runs -Directory | ?{(dir $_.FullName | measure).Count -eq 0} | %{echo "rmdir: $_"; ri $_.FullName} # delete empty folders
echo 'Done.'
