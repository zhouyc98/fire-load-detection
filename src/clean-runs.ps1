cd $PSScriptRoot

$fs=dir ./runs -File -Recurse -Exclude *.ps1 | ?{$_.Length -lt 30000} # files that size < 3KB
$ds=dir ./runs -Directory | ?{(dir $_.FullName | measure).Count -eq 0} # empty folders

if($fs -and $ds){
	$fs | %{echo "rm:    $_"}
	$ds | %{echo "rmdir: $_"}

	$confirmation = Read-Host "`nSure to delete (donot delete running files)? y/[n]"
	if ($confirmation -eq 'y') {
		$fs | ri
		$ds | ri
		echo 'Done.'
	}
}
