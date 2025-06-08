#!/bin/bash

# Get the input parameters
trip_duration_days=$1
miles_traveled=$2
total_receipts_amount=$3

# Call the Python script and get the result
python3 -c "from template import calculate_reimbursement; print(calculate_reimbursement($trip_duration_days, $miles_traveled, $total_receipts_amount))" 