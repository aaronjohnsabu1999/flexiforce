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
        self.force_path = repo_path + base_path + 'robot_force.sto'
        
        #Model Initialization/preparation
        self.model = osim.Model(repo_path + base_path + model)
        self.model.setup()
        self.state = self.model.initSystem()

        #Generate Elbow Trajectory from desired characteristics
        self.t = np.linspace(0, curl_time, int(curl_time*sps))
        k = 12/curl_time
        b = curl_time/2
        self.elbow_trajectory = np.deg2rad(rom)/(1 + np.exp(-k*(self.t-b)))

        #Initialize trajectory and forces dictionaries
        self.traj_dict = {
            "r_shoulder_elev": np.zeros(self.t.shape),
            "r_elbow_flex": self.elbow_trajectory
        }
        self.force_dict = {
            "Fz": np.zeros(self.t.shape)
        }
        
        #Create Tables
        self.traj_table = self.make_OSIMtable(self.traj_dict)
        self.force_table = self.make_OSIMtable(self.force_dict)

        #Convert Table objects to .sto
        self.sto = osim.STOFileAdapter()
        self.sto.write(self.traj_table, self.traj_path)
        self.sto.write(self.force_table, self.force_path)

        #Initialize Static Optimizer/Tools
        self.optimizer = osim.StaticOptimization()
        self.analyzer = osim.AnalyzeTool()
        self.analyzer.setModel(self.model)
        self.analyzer.setModelFilename(self.model.getDocumentFileName())
        self.analyzer.setCoordinatesFileName(self.traj_path)
        #self.analyzer.setExternalLoadsFileName(self.force_path)
        self.analyzer.setStatesFromMotion(self.state, osim.Storage(self.traj_path), True)
        self.analyzer.setName("Bicep_Curl")

        

    def step_simulation(self, step_index, force_vector):
        '''
        Perform static optimization for a single time index under an external force

        step_index: the time index to perform static optimization
        force_vector: the external force under which the exercise is occurring

        returns: computed biceps activation for this external force
        '''
        #Set time of step in Static Optimizer/Tools
        t = self.t[step_index]
        self.optimizer.setStartTime(t)
        self.optimizer.setEndTime(t)
        self.analyzer.setStartTime(t)
        self.analyzer.setFinalTime(t)

        #add force_vector to time index
        robot_force = osim.TimeSeriesTable(self.force_path)
        Fz = robot_force.getDependentColumn('Fz').to_numpy()
        Fz[step_index] = force_vector
        force_dict = {
            "Fz": Fz
        }
        self.force_table = self.make_OSIMtable(force_dict)
        self.sto.write(self.force_table, self.force_path)

        #Setup step analyzer
        self.step_analysis = self.analyzer.clone()
        self.step_analysis.updAnalysisSet().cloneAndAppend(self.optimizer)
        self.step_analysis.setResultsDir(self.repo_path + self.base_path + "Results")
        self.step_analysis.printToXML(self.base_path + "static_optimization_setup.xml")

        #Run
        self.step_analysis = osim.AnalyzeTool(self.base_path + "static_optimization_setup.xml", True)
        self.step_analysis.run()

        #Access Results to return biceps activation
        results = osim.TimeSeriesTable(self.base_path + "Results\Bicep_Curl_StaticOptimization_activation.sto")
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
    


    def _reset(self):
        '''
        Reset the external force to zero to enable simulation rerun without reinitialization of a new object
        '''
        for key, _ in self.force_dict.item():
            self.force_dict[key] = np.zeros(self.t.shape)
        self.force_table = self.make_OSIMtable(self.force_dict)
        self.sto.write(self.force_table, self.force_path)



    def _traj(self):
        '''
        Provide the elbow trajectory for convenience
        '''
        return self.t, self.elbow_trajectory