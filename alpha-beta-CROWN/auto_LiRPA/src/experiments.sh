#!/bin/bash

# Arrays of values for k, epsilon, p, and image number
k_values=(15)
epsilon_values=(0.20 0.25)
p_values=(2 inf)
image_numbers=(6 7 8)

# Loop over all combinations of k, epsilon, p, and image number
for k in "${k_values[@]}"; do
    for epsilon in "${epsilon_values[@]}"; do
        for p in "${p_values[@]}"; do
            for image_number in "${image_numbers[@]}"; do
                # Construct the image path
                image_path="../images/mnist_${image_number}.png"
                
                # Check if the image file exists
                if [[ ! -f "$image_path" ]]; then
                    echo "Error: Image file $image_path does not exist."
                    continue
                fi
                
                # Construct the log file name
                log_file="../logs/results_k${k}_epsilon${epsilon}_p${p}_image${image_number}.log"
                
                # Run the command and log the results
                if python main.py -k "$k" --epsilon "$epsilon" -p "$p" --image-path "$image_path" > "$log_file"; then
                    echo "Success: Command succeeded for k=$k, epsilon=$epsilon, p=$p, image_number=$image_number"
                else
                    echo "Error: Command failed for k=$k, epsilon=$epsilon, p=$p, image_number=$image_number"
                fi
            done
            done
        done
    done
done
