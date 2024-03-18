from SystemSetup_module import SystemSetup
import csv

# Initialize SystemSetup and start a session
optical_system = SystemSetup()
optical_system.start_session()

# Set global parameters
optical_system.set_wavelengths([486, 587, 656])
optical_system.set_fd(5)
optical_system.set_dimensions('m')
optical_system.set_fields([(0, 0.7)])


# Store tabulated merit function values
tabulated_merit_functions = []

################################# PAPER STARTING POINTS #################################


triplets_data = [
    ("FK51", "638424.320", "SF1", 0.78040, -0.26833, -0.28528, -2.09468, 1.4514),
    ("FK51", "638424.320", "SF10", 0.74376, -0.27169, -0.28857, -2.00724, 1.6283),
    ("FK51", "638424.320", "SF4", 0.71635, -0.27411, -0.29081, -1.90816, 1.8667),
    ("FK51", "638424.320", "SF8", 0.88060, -0.26021, -0.27696, -2.10795, 1.1926),
    ("FK51", "KZFS4", "SF2", 1.11226, -0.22997, -0.24275, -1.9530, 0.98067),
    ("FK51", "KZFS4", "SF57", 0.66332, -0.25813, -0.27093, -1.65476, 3.0816),
    ("FK51", "KZFS5", "SF10", -1.34126, -0.16583, -0.17521, -0.52381, 0.79746),
    ("FK51", "KZFS5", "SF57", 0.91218, -0.23371, -0.25026, -1.56151, 1.4607),
    ("FK51", "LAK10", "SF66", 0.64693, -0.22008, -0.23801, -0.77359, 7.0171),
    ("FK51", "LAK10", "SF67", 0.65421, -0.21678, -0.23460, -0.76903, 6.2693)
]


# Create surfaces with placeholder values outside the loop
num_surfaces = 6  # Total number of surfaces for a triplet system
for num in range(1, num_surfaces + 1):
    thickness = 0.1 if num % 2 != 0 else 0  # Assign thickness 0.1 to glass surfaces, 0 to air gaps
    optical_system.Surface(optical_system, number=num, radius=1, thickness=thickness, material=None)

# Loop over each triplet to update the system and test the merit function
for index, triplet in enumerate(triplets_data):
    glass1, glass2, glass3, r1, r1_prime, r2, r2_prime, r3 = triplet
    
    # Update the surfaces with the actual triplet data
    optical_system.surfaces[1].set_parameters(radius=r1, thickness=0.1, material=glass1)
    optical_system.surfaces[2].set_parameters(radius=r1_prime, thickness=0)
    optical_system.surfaces[3].set_parameters(radius=r2, thickness=0.1, material=glass2)
    optical_system.surfaces[4].set_parameters(radius=r2_prime, thickness=0)
    optical_system.surfaces[5].set_parameters(radius=r3, thickness=0.1, material=glass3)
    optical_system.surfaces[6].set_parameters(radius=1e10, thickness=0)  # r3' is infinity

    optical_system.set_paraxial_image_distance()

    # Make all thicknesses variable and optimize the system
    optical_system.make_all_thicknesses_variable(last_one = False)
    optical_system.make_all_radii_variable()
    optical_system.optimize_system(efl=1, mxt=0.1) 

    # Save the system with a unique file path for each triplet
    file_path = f"C:/CVUSER/triplet_system_{index+1}"  # Unique file path for each system
    optical_system.save_system(file_path, seq=True)

    # Calculate the merit function
    tabulated_merit = optical_system.error_fct(efl=1)
    tabulated_merit_functions.append((glass1, glass2, glass3, tabulated_merit))




################################# NOW COMPARE WITH OUR STARTING POINT #################################
    

# Comparison results
comparison_results = []


# Assuming we have a list of .seq file paths and a dictionary mapping lens numbers to materials
seq_file_paths = ["C:/CVUSER/FinalOptimized_Node15.seq", "C:/CVUSER/FinalOptimized_Node16.seq", "C:/CVUSER/FinalOptimized_Node12.seq", "C:/CVUSER/FinalOptimized_Node10.seq", "C:/CVUSER/FinalOptimized_Node9.seq"] 


# Loop over each triplet material combination
for triplet in triplets_data:
    glass1, glass2, glass3, r1, r1_prime, r2, r2_prime, r3 = triplet

    # Loop over each seq file path
    for seq_file_path in seq_file_paths:
        # Load the .seq file into the system
        load_command = f'run "{seq_file_path}"; GO'
        optical_system.cv.Command(load_command)
        
        # Change the material of the three lenses using the current triplet combination
        optical_system.cv.Command(f"GL1 S1 {glass1}")
        optical_system.cv.Command(f"GL1 S3 {glass2}")
        optical_system.cv.Command(f"GL1 S5 {glass3}")
        
        # Make all thicknesses and radii variable
        for i in range(1, 6):
            optical_system.cv.Command(f"CCY S{i} 0")    
            i+=1

        for i in range(1, 6):
            optical_system.cv.Command(f"THC S{i} 0")

        # Fix all materials
        for i in range(1, 6):
            optical_system.cv.Command(f"GC1 S{i} 100")

        # Perform optimization
        optical_system.optimize_system(efl=1, mnt=0.02)  # Replace 'efl=1' with the desired effective focal length
        
        # Calculate the optimized merit function
        optimized_merit = optical_system.error_fct(efl=1)
        
        # Record the comparison
        comparison_results.append({
            'Seq_File_Path': seq_file_path,
            'Triplet': f"{glass1}_{glass2}_{glass3}",
            'Optimized_Merit': optimized_merit,
        })

         # Save the system with a new file name indicating the materials used
        base_name = seq_file_path.split('/')[-1].split('.')[0]  # Extract base name of the file without extension
        optimized_file_path = f"C:/CVUSER/{base_name}_optimized_{glass1}_{glass2}_{glass3}.seq"
        optical_system.save_system(optimized_file_path, seq=True)

# Save the comparison results to a CSV
csv_file_path = "comparison_results.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=comparison_results[0].keys())
    writer.writeheader()
    writer.writerows(comparison_results)

# Stop session after all files have been processed
optical_system.stop_session()



# Prepare the final structure for the CSV export
final_comparison_results = []

# Iterate over each unique triplet to gather its comparison results
for glass_combination in tabulated_merit_functions:
    glass1, glass2, glass3, tabulated_merit = glass_combination
    triplet_identifier = f"{glass1}_{glass2}_{glass3}"

    # Find all optimized merits for this triplet from different starting points
    optimized_merits = [comp for comp in comparison_results if comp['Triplet'] == triplet_identifier]

    # For each starting point's optimized merit, prepare a row in the final comparison results
    for optimized_merit in optimized_merits:
        final_comparison_results.append({
            'Glass Combination': f"{glass1}, {glass2}, {glass3}",
            'Tabulated Merit Function': tabulated_merit,
            'Seq File Path': optimized_merit['Seq_File_Path'],
            'Optimized Merit Function': optimized_merit['Optimized_Merit']
        })

# CSV headers
csv_headers = ['Glass Combination', 'Tabulated Merit Function', 'Seq File Path', 'Optimized Merit Function']

# Export to CSV
csv_file_path = "final_comparison_results.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=csv_headers)
    writer.writeheader()
    writer.writerows(final_comparison_results)
