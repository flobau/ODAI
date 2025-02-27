from SystemSetup_module import SystemSetup
from SystemNode_module import SystemNode, SystemTree

class OpticalSystemManager:
    def __init__(self):
        # Default parameters
        self.default_wavelengths = [486.1327, 546.074, 587.5618, 632.2, 657.2722]
        self.default_fd = 5
        self.default_dimensions = 'm'
        self.default_fields = [(0, 3), (0, 6), (0, 35)]
        self.default_efl = 1

        # Root parameters
        self.epsilon = 1
        self.lens_thickness_steps = [0.05, 0.1, 0.15, 0.4]
        self.air_distance_steps = [0]
        self.lens_thickness = 0.2

        # Initial system parameters
        self.surface1_radius = 59.33336
        self.surface1_thickness = 0.2
        self.surface1_material = "NBK7_SCHOTT"
        self.surface2_radius = -391.44174
        self.surface2_thickness = 97.703035

        self.base_file_path = "C:/CVUSER"
        self.target_depth = 4
        self.starting_depth = 0

        # Initialize optical system
        self.optical_system = SystemSetup()
        self.system_tree = None

    def start_system(self):
        self.optical_system.start_session()
        self.optical_system.create_new_system()

        # Set default parameters
        self.optical_system.set_wavelengths(self.default_wavelengths)
        self.optical_system.set_fd(self.default_fd)
        self.optical_system.set_dimensions(self.default_dimensions)
        self.optical_system.set_fields(self.default_fields)

        # Create initial optical system
        self._create_initial_system()

        # Optimize the initial system
        self.optical_system.optimize_system(self.default_efl, constrained=False)
        self.optical_system.update_all_surfaces_from_codev(debug=False)

        # Save initial system parameters and create system tree
        self._create_system_tree()

    def _create_initial_system(self):
        surface1 = self.optical_system.Surface(self.optical_system, 1, self.surface1_radius, self.surface1_thickness, self.surface1_material)
        surface1.make_radius_variable()
        surface2 = self.optical_system.Surface(self.optical_system, 2, self.surface2_radius, self.surface2_thickness)
        surface2.make_radius_variable()

        self.optical_system.set_paraxial_image_distance()

    def _create_system_tree(self):
        root_system = self.optical_system.save_system_parameters()
        root_merit = self.optical_system.error_fct(self.default_efl, constrained=False)
        root_efl = self.optical_system.get_efl_from_codev()

        root_params = {
            'epsilon': self.epsilon,
            'lens_thickness_steps': self.lens_thickness_steps,
            'air_distance_steps': self.air_distance_steps,
            'lens_thickness': self.lens_thickness,
        }

        root_node = SystemNode(system_params=root_params, optical_system_state=root_system, merit_function=root_merit, efl=root_efl, is_optimized=True)
        self.system_tree = SystemTree(root_node)

    def evolve_and_optimize(self):
        self.optical_system.evolve_optimized_systems(self.system_tree, self.starting_depth, self.target_depth, self.base_file_path, self.default_efl)
        self.system_tree.print_tree()
        self.system_tree.final_optimization(self.optical_system, self.default_efl, self.base_file_path)
        self.system_tree.print_final_optimized_systems_table()
        self.system_tree.plot_optimization_tree()

    def end_system(self):
        self.optical_system.stop_session()

    # Methods for updating initial system parameters
    def set_initial_system_parameters(self, surface1_radius, surface1_thickness, surface1_material, surface2_radius, surface2_thickness):
        self.surface1_radius = surface1_radius
        self.surface1_thickness = surface1_thickness
        self.surface1_material = surface1_material
        self.surface2_radius = surface2_radius
        self.surface2_thickness = surface2_thickness

    # Methods for updating root parameters
    def set_root_parameters(self, epsilon, lens_thickness_steps, air_distance_steps, lens_thickness):
        self.epsilon = epsilon
        self.lens_thickness_steps = lens_thickness_steps
        self.air_distance_steps = air_distance_steps
        self.lens_thickness = lens_thickness

# Usage example
optical_system_manager = OpticalSystemManager()
optical_system_manager.set_initial_system_parameters(59.33336, 0.2, "NBK7_SCHOTT", -391.44174, 97.703035)
optical_system_manager.set_root_parameters(1, [0.05, 0.1, 0.15, 0.2], [0], 0.2)
optical_system_manager.start_system()
optical_system_manager.evolve_and_optimize()
optical_system_manager.end_system()