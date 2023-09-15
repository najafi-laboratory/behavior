# Specify the local directory containing the files to upload
$localDirectory = "" #REPLACE WITH YOUR LOCAL DIRECTORY

# Specify the Dropbox folder where you want to upload the files
$dropboxFolder = "" #REPLACE WITH DROPBOX DIRECTORY

#List files in local directory
$files = Get-ChildItem $localDirectory

# Loop through the files and copy them to Dropbox
foreach ($file in $files) {
    $destinationPath = Join-Path -Path $dropboxFolder -ChildPath $file.Name
    Copy-Item -Path $file.FullName -Destination $destinationPath -Force -Recurse
    Write-Host "Uploaded $($file.Name) to Dropbox"
}
#executes using: .\upload_to_dropbox.ps1
