#Specify Local Endpoint UUID and Directory, currently Rikhil's Laptop
$localEndpoint = "d269fe7a-5898-11ee-876b-1dc3121de006:/C/Users/rikhi/research/data"

#Specify Globus endpoint and Directory
$globusEndppoint = "6df312ab-ad7c-4bbc-9369-450c82f0cb92:/storage/coda1/p-fnajafi3/0/shared/"

globus transfer -s exists -r $localEndpoint $globusEndppoint