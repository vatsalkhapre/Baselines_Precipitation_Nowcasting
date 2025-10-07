

def save_rainydaysdata_file(mode, data_dir, method = None, csv_file_add = None):
    if mode == "Naive":
        if method == 0:
            grouped_files = rainy_days(mode, data_dir)
            print(grouped_files)
            
            with open('rainy_days.pkl', 'wb') as f:
                pickle.dump(grouped_files, f)
            print("Rainydays file saved")

        elif method == 1:
            grouped_files = rainy_days_plus_prev_days(mode, data_dir)
            with open('/home/vatsal/MOSDAC/rainy_days_plusprev_days.pkl', 'wb') as f:
                pickle.dump(grouped_files, f)
            print("Rainydays with previous days file saved")
    
    elif mode == 'IMD':
        grouped_files = rainy_days(mode, data_dir, csv_file_add)
        print(grouped_files)
        
        with open('rainy_days_IMD.pkl', 'wb') as f:
            pickle.dump(grouped_files, f)
        print("Rainydays file saved")

def _ensure_npy_path(data_dir, filename):
    # Accepts filename either with or without .npy, returns full path
    if filename.lower().endswith('.npy'):
        return os.path.join(data_dir, filename)
    else:
        return os.path.join(data_dir, filename + '.npy')

# ============================== Rainy days (Above 1 or 2 std)========================================

def rainy_days(mode, data_folder_path = None, csv_file_add = None):

    ####### Naive Logic #######

    """
    Mode decides the selection of the rainy days based on either the files or the csv.
    """
    grouped_files = defaultdict(list)
    spatial_means = defaultdict(list)
    # num_true_values = np.sum(mask.values)  # Count of True values


    # Iterate through sorted files in the folder
    for filename in sort_files(data_folder_path):
        date_part = filename[:9]  # Extract date (DDMMMYYYY)
        file_path = os.path.join(data_folder_path, filename)

        # Open dataset and process the DBZ variable
        try:
            data_array = np.load(file_path, allow_pickle=True)
            
            data_array = preprocessing_radar(data_array)  # Apply preprocessing
            
            # print(spatial_mean)
            # Store results
            grouped_files[date_part].append(filename)
            

            if mode == "files":
                spatial_mean = np.mean(data_array)  # Compute spatial mean
                spatial_means[date_part].append(spatial_mean)

        except KeyError as e:
            print(f"KeyError: {e} in file {filename}. Skipping this file.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}. Skipping this file.")
        

    # Convert defaultdicts to regular dictionaries
    grouped_files = dict(grouped_files)

    
    rainy_days_ = defaultdict(list)

    if mode == "Naive":

        # Identify rainy days based on threshold

        spatial_means = dict(spatial_means)
        flattened_sp_means = np.concatenate(list(spatial_means.values()))

        # Flatten all spatial means and compute global statistics
    
        mean_of_means = np.mean(flattened_sp_means)
        std_of_means = np.std(flattened_sp_means)

        print(f"Overall Mean: {mean_of_means}")
        print(f"Overall Standard Deviation: {std_of_means}")

        for date, mean_values in spatial_means.items():
            if np.mean(mean_values) > mean_of_means:  # Threshold: Mean of all spatial means
                rainy_days_[date] = grouped_files[date]

    ##################FROM IMD###############

    elif mode == "IMD":
        print("IMD")
        df = pd.read_csv(csv_file_add)
        days_lis = []
        for col in list(df.columns):
            days_lis.append(df[col].values)
        flattened_days = [convert_date(str(date)) for arr in days_lis for date in arr if pd.notna(date)]
        for date in flattened_days:
            print(date)
            if date in grouped_files.keys():
                print("Date Present")
                rainy_days_[date] = grouped_files[date]


    return dict(rainy_days_)


# ============================== Rainy days (Above 1 or 2 std) including previous days========================================

def rainy_days_plus_prev_days(folder_path):
    

    grouped_files = defaultdict(list)
    spatial_means = defaultdict(list)

    # Iterate through sorted files in the folder
    for filename in sort_files(folder_path):
        date_part = filename[:9]  # Extract date (DDMMMYYYY)
        
        file_path = os.path.join(folder_path, filename)

        # Open dataset and process the DBZ variable
        try:
            data_array = np.load(file_path, allow_pickle=True)
            data_array = preprocessing_radar(data_array)  # Apply preprocessing

            spatial_mean = np.mean(data_array)  # Compute spatial mean

            # Store results
            grouped_files[date_part].append(filename)
            spatial_means[date_part].append(spatial_mean)

        except KeyError as e:
            print(f"KeyError: {e} in file {filename}. Skipping this file.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}. Skipping this file.")
        # finally:
        #     # Ensure dataset is closed to free memory
        #     dataset.close()

    # Convert defaultdicts to regular dictionaries
    grouped_files = dict(grouped_files)
    spatial_means = dict(spatial_means)

    # Flatten all spatial means and compute global statistics
    flattened_sp_means = np.concatenate(list(spatial_means.values()))
    mean_of_means = np.mean(flattened_sp_means)
    std_of_means = np.std(flattened_sp_means)

    print(f"Overall Mean: {mean_of_means}")
    print(f"Overall Standard Deviation: {std_of_means}")

    # Identify rainy days based on threshold
    rainy_days_ = defaultdict(list)
    for date, mean_values in spatial_means.items():
        if np.mean(mean_values) > mean_of_means:  # Threshold: Mean of all spatial means
            rainy_days_[date] = grouped_files[date]

            prev_date_val = int(date[:2])-1
            month_year = date[2:]
            prev_date = f"{prev_date_val:02d}{month_year}"

            if prev_date in grouped_files and prev_date not in rainy_days_:
                rainy_days_[prev_date] = grouped_files[prev_date]


    print("Rainy Days:")
    for date, files in rainy_days_.items():
        print(f"{date}: {files}")

    return dict(rainy_days_)
