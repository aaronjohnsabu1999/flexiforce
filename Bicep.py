import opensim as osim
import numpy as np

class Bicep_Curl: 

    def __init__(self, curl_time = 2, rom = 135, sps = 250, base_path = 'OpenSIM_utils\\Arm26\\Flexiforce\\', model = 'arm26.osim', repo_path = "C:\\Users\\saman\\OneDrive\\Documents\\GitHub\\flexiforce\\"):
        '''
        curl_time: duration of the biceps curl in seconds
        rom: range of motion of the elbow joint; ie: following a trajectory from 0 degrees to rom degrees
        sps: steps per second, defines the resolution of the simulation

        base_path: path to the directory with the .osim model and where static optimization results will be stored
        model: .osim filename of the model in the base_path directory to be used for the analysis
        repo_path: the absolute path to the location where the github repo code is stored on your machine
        '''
        
        #Housekeeping
        self.base_path = base_path
        self.repo_path = repo_path
        self.traj_path = repo_path + base_path + 'bicep_curl.sto'
        self.force_path_sto = repo_path + base_path + 'robot_force.sto'
        self.force_path_xml = repo_path + base_path + 'robot_force.xml'
        
        #Model Initialization/preparation
        self.model = osim.Model(repo_path + base_path + model)
        self.model.setup()

        #Generate Elbow Trajectory from desired characteristics
        self.t = np.linspace(0, curl_time, int(curl_time*sps))
        k = 12/curl_time
        b = curl_time/2
        self.elbow_trajectory = np.deg2rad(rom)/(1 + np.exp(-k*(self.t-b)))

        #Initialize trajectory 
        self.traj_dict = {
            "r_shoulder_elev": np.zeros(self.t.shape),
            "r_elbow_flex": self.elbow_trajectory
        }
        self.traj_table = self.make_OSIMtable(self.traj_dict)
    
        #Convert Table objects to .sto
        self.sto = osim.STOFileAdapter()
        self.sto.write(self.traj_table, self.traj_path)

        #Initialize Static Optimizer/Tools
        self.optimizer = osim.StaticOptimization()
        self.analyzer = osim.AnalyzeTool()
        self.analyzer.setModel(self.model)
        self.analyzer.setModelFilename(self.model.getDocumentFileName())
        self.analyzer.setCoordinatesFileName(self.traj_path)
        self.analyzer.setName("Bicep_Curl")

        #Initialize External Load
        x = np.zeros(self.t.shape)
        y = np.zeros(self.t.shape)
        z = np.zeros(self.t.shape)
        fx = np.zeros(self.t.shape)
        fy = np.zeros(self.t.shape)
        fz = np.zeros(self.t.shape)
        self.ext_load = {
            "r_ulna_radius_hand_force_vx": fx,
            "r_ulna_radius_hand_force_vy": fy,
            "r_ulna_radius_hand_force_vz": fz,
            "r_ulna_radius_hand_force_px": x, 
            "r_ulna_radius_hand_force_py": y, 
            "r_ulna_radius_hand_force_pz": z,
        }
        
        
        

    def step_simulation(self, step_index, force_vector):
        '''
        Perform static optimization for a single time index under an external force

        step_index: the time index to perform static optimization
        force_vector: the external force under which the exercise is occurring

        returns: computed biceps activation for this external force
        '''
        #Set time of step in Static Optimizer/Tools
        self.step_index = step_index
        t = self.t[step_index]
        self.optimizer.setStartTime(t)
        self.optimizer.setEndTime(t)
        self.analyzer.setStartTime(t)
        self.analyzer.setFinalTime(t)
    
        # Create State object from the shoulder and elbow coordinates at the step_index
        data = osim.Vector.createFromMat(np.asarray([0.0, self.elbow_trajectory[step_index]]))
        state = self.model.initSystem()
        state.setQ(data)
        self.model.assemble(state)

        #Get Transformation Matrix between ground and r_ulna_radius_hand frames
        for frame in self.model.getFrameList():
            if frame.getName() != "r_ulna_radius_hand":
                continue
            T_GR = frame.getTransformInGround(state)

        #Get contact point position in r_ulna_radius_hand frame
        for ctc_geom in self.model.get_ContactGeometrySet():
            if ctc_geom.getName() != "Robot_Contact":
                continue
            V_R = ctc_geom.getLocation()

        #set position of contact geometry in ground frame during step_index
        V_G = T_GR.shiftFrameStationToBase(V_R)
        self.ext_load["r_ulna_radius_hand_force_px"][step_index] = V_G[0]
        self.ext_load["r_ulna_radius_hand_force_py"][step_index] = V_G[1]
        self.ext_load["r_ulna_radius_hand_force_pz"][step_index] = V_G[2]
        
        #Set Robot Force during step_index, make XML
        self.ext_load["r_ulna_radius_hand_force_vx"][step_index] = force_vector[0]
        self.ext_load["r_ulna_radius_hand_force_vy"][step_index] = force_vector[1]
        self.ext_load["r_ulna_radius_hand_force_vz"][step_index] = force_vector[2]
        self.make_XML(self.ext_load)
        
        #Setup step analyzer
        self.step_analysis = self.analyzer.clone()
        self.step_analysis.updAnalysisSet().cloneAndAppend(self.optimizer)
        self.step_analysis.setExternalLoadsFileName(self.force_path_xml)
        #self.step_analysis.setExternalLoads(self.force_path_xml)
        self.step_analysis.setResultsDir(self.repo_path + self.base_path + "Results")
        self.step_analysis.printToXML(self.base_path + "static_optimization_setup.xml")

        #Run
        self.step_analysis = osim.AnalyzeTool(self.base_path + "static_optimization_setup.xml", True)
        self.step_analysis.run()

        #Access Results to return biceps activation
        results = osim.TimeSeriesTable(self.base_path + "Results\\Bicep_Curl_StaticOptimization_activation.sto")
        biceps_activation = results.getDependentColumn('BIClong').to_numpy()

        return biceps_activation



    def make_OSIMtable(self, columns = dict):
        '''
        Helper function to make table objects for conversion to OpenSIM native file format .sto

        columns: dictionary with the dependent columns for the table object, independent column is self.t (time) as specified by init
        '''
        OSIMtable = osim.TimeSeriesTable(self.t)
        for key, value in columns.items():
            OSIMtable.appendColumn(key, osim.Vector.createFromMat(value))

        return OSIMtable
    

    def make_XML(self, columns = dict):
        '''
        Helper function to make .xml & .sto files for the external load forces

        columns: dictionary with the dependent columns for the table object, independent column is self.t (time) as specified by init
        '''
        #create .sto files with the force
        OSIMtable = osim.TimeSeriesTable(self.t)
        for key, value in columns.items():
            OSIMtable.appendColumn(key, osim.Vector.createFromMat(value))
        self.sto.write(OSIMtable, self.force_path_sto)
        
        #Create ExternalForce & set XML file
        force = osim.ExternalForce()
        force.set_data_source_name(self.force_path_sto)
        force.set_force_identifier("r_ulna_radius_hand_force_v")
        force.set_point_identifier("r_ulna_radius_hand_force_p")
        force.set_force_expressed_in_body("ground")
        force.set_point_expressed_in_body("ground")
        force.set_appliesForce(True)
        
        for body in self.model.getBodyList():
            if body.getName() != "r_ulna_radius_hand":
                continue
            force.set_applied_to_body(body.getName())
        
        #load = osim.ExternalLoads()
        #load.setDataFileName(self.force_path_sto)
        #load.addComponent(force)
        
        #self.model.addForce(force)
        #self.model.finalizeConnections()
        
        #load.printToXML(self.force_path_xml)
        force.printToXML(self.force_path_xml)
        #self.model.initSystem()
        #self.model.addForce(force)
        #self.model.finalizeConnections()



    def _traj(self):
        '''
        Provide the elbow trajectory for convenience
        '''
        return self.t, self.elbow_trajectory