import os
import dropbox
import schedule
import time

# Dropbox access token (replace with your own)
DROPBOX_ACCESS_TOKEN = 'sl.BlmbifYr270-bsYt_LVLXcab9eKlVvr4MUtK_D9aGZjxvLeO-wevaL3MnKXHXJIcOBUzdqczhhe5xM-5sgvyXwOiImfDcqmvJbq800WwFbgJmCj8G7xtlm_a2RwOT2qKgfqo-xi6HvBtD-obQnnGMrk'

# Local directory containing files to copy 
LOCAL_DIRECTORY = '' #INSERT PATH HERE

# Dropbox directory where files will be copied
DROPBOX_DIRECTORY = '/BioSci-Najafi'

# Initialize the Dropbox client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def copy_files_to_dropbox():
    print("Executing!")
    for root, dirs, files in os.walk(LOCAL_DIRECTORY):
        for file in files:
            local_path = os.path.join(root, file)
            dropbox_path = os.path.join(DROPBOX_DIRECTORY, os.path.relpath(local_path, LOCAL_DIRECTORY))

            with open(local_path, 'rb') as f:
                dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.add)
    print('Successfully copied files!')

def main():
    # Initial copy
    copy_files_to_dropbox()

    #Schedule the script to run every hour
    schedule.every(1).hours.do(copy_files_to_dropbox)

    #flag to stop the schedule
    stop_schedule = False
    while not stop_schedule:
        schedule.run_pending()
        #print("pending!") for debugging
        time.sleep(1)

if __name__ == "__main__":
    main()