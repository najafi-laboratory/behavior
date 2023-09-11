# Specify the local directory containing the files to upload
$localDirectory = "C:\path\to\local\directory"

# Specify the Dropbox folder where you want to upload the files
$dropboxFolder = "/BioSci-Najafi"

# Path to the Dropbox CLI executable
$dropboxCLI = "C:\Program Files\Dropbox\Client\Dropbox.exe"

# Dropbox access token
$dropboxAccessToken = "sl.Bl4L7XvyXjMLZFig92k-4bD26nVliefbUPTyVniVIZTFugDNP72UYwZBq1IcSEFn2DjZFd8QralEpBDBPhhzT_XxQMg7InpNbDwg5sUSrxBKxUxJPbu9-iw_EwMvNejvPVBziCEXd_Lk8EwMEfvlGb0"

# Upload files to Dropbox
& $dropboxCLI upload $localDirectory $dropboxFolder

Write-Host "Files uploaded to Dropbox at $(Get-Date)"


'''
# List files in the local directory
$files = Get-ChildItem $localDirectory

# Loop through the files and copy them to Dropbox
foreach ($file in $files) {
    $destinationPath = Join-Path -Path $dropboxDirectory -ChildPath $file.Name
    Copy-Item -Path $file.FullName -Destination $destinationPath -Force
    Write-Host "Uploaded $($file.Name) to Dropbox"
}
'''
#executes using: .\upload_to_dropbox.ps1