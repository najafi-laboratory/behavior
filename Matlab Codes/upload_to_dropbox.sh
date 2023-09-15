#!/bin/bash

# Set the local directory path
LOCAL_DIRECTORY="YOUR_LOCAL_DIRECTORY"

# Set the Dropbox directory path
DROPBOX_DIRECTORY="YOUR_DROPBOX_DIRECTORY"

# Change to the local directory
cd "$LOCAL_DIRECTORY" || exit

# Upload files to Dropbox using Dropbox Uploader
~/dropbox_uploader.sh upload . "$DROPBOX_DIRECTORY"

echo "Files uploaded to Dropbox at $(date)"

#executes using chmod +x upload_to_dropbox.sh
