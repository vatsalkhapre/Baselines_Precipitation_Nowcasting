import pickle

def check_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Contents of the .pkl file:")
            for key, val in data.items():
                print(key)
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = "/home/vatsal/Nowcasting/Baselines/Precip_nowcast_code/Rainy_days_file/rainydays_above-1std_IMD.pkl"  # Replace with the actual path to your .pkl file
check_pkl_file(file_path)