from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator
import yaml
from pathlib import Path


# ==============================================================================
# DATACLASSES: Think of these like blueprints for objects that hold data.
# The @dataclass decorator automatically creates __init__, __repr__, etc.
# ==============================================================================

@dataclass
class EnsembleMember:
    """
    Represents an ensemble member configuration for ONE model.
    
    This stores the data from models.yaml for a single model.
    Instead of writing __init__ manually, @dataclass does it for us.
    
    Example: For AWI-CM-1-1-MR, this would store:
        N_members = 5
        active = True
        ensemble_members = ['AWI-CM-1-1-MR_r1i1p1f1', 'AWI-CM-1-1-MR_r2i1p1f1', ...]
        primary_member = 'AWI-CM-1-1-MR_r1i1p1f1'
    """
    N_members: int              # Number of ensemble members
    ensemble_members: List[str] # List of all member names
    primary_member: str         # The primary member to use


@dataclass
class ModelVariable:
    """
    Represents a model's status for a specific variable.
    
    This stores the data from variable_status.yaml for ONE model 
    and ONE variable combination.
    
    Example: For AWI-CM-1-1-MR and tas_annual_max, this would store:
        active = True
        failure_mode = None
        invalid_years = []
        valid_years = [1979, 1980, ..., 2024]
    """
    active: bool                    # Is this model active for this variable?
    failure_mode: Optional[str]     # Why it failed (None if no failure)
    invalid_years: List[int]        # Years with bad data
    valid_years: List[int]          # Years with good data


@dataclass
class ModelForAnalysis:
    """
    A convenient package of ALL the info you need to analyze one model.
    
    This combines data from BOTH yaml files into one object that has
    everything you need: the model name, which members to use, and 
    which years are valid.
    
    This is what you'll actually work with in your analysis loops.
    """
    name: str                   # Model name (e.g., 'AWI-CM-1-1-MR')
    primary_member: str         # The primary member name
    all_members: List[str]      # All ensemble member names
    valid_years: List[int]      # Years you can analyze
    failure_mode: Optional[str] # Any failure info
    
    def get_primary(self) -> str:
        """Get just the primary member"""
        return self.primary_member
    
    def get_all_members(self) -> List[str]:
        """Get all ensemble members"""
        return self.all_members


@dataclass
class ClimateModelConfig:
    """
    Main configuration class that holds ALL models from BOTH yaml files.
    
    This is the "brain" that:
    1. Loads both yaml files
    2. Stores all the data
    3. Lets you query which models are active for a variable
    4. Packages up the info you need for analysis
    """
    
    # field(default_factory=dict) creates an empty dict when the object is created
    # ensemble_config will store: {'AWI-CM-1-1-MR': EnsembleMember(...), ...}
    ensemble_config: Dict[str, EnsembleMember] = field(default_factory=dict)
    
    # variable_status will store: {'tas_annual_max': {'AWI-CM-1-1-MR': ModelVariable(...), ...}, ...}
    variable_status: Dict[str, Dict[str, ModelVariable]] = field(default_factory=dict)
    
    @classmethod
    def from_yaml_files(cls, ensemble_path: str, variable_path: str) -> 'ClimateModelConfig':
        """
        Load configuration from two YAML files.
        
        @classmethod means this is called on the CLASS, not an instance:
            config = ClimateModelConfig.from_yaml_files('models.yaml', 'variable_status.yaml')
        
        Instead of:
            config = ClimateModelConfig()  # would create empty config
        
        Args:
            ensemble_path: Path to models.yaml
            variable_path: Path to variable_status.yaml
            
        Returns:
            A fully loaded ClimateModelConfig object
        """
        # Create an empty ClimateModelConfig object
        config = cls()
        
        # === LOAD ENSEMBLE CONFIGURATION (models.yaml) ===
        with open(ensemble_path, 'r') as f:
            ensemble_data = yaml.safe_load(f)  # Reads yaml into a Python dict
            
            # ensemble_data looks like: {'models': {'AWI-CM-1-1-MR': {...}, ...}}
            # Loop through each model in the 'models' section
            for model_name, model_data in ensemble_data['models'].items():
                # model_data is a dict like: {'N_members': 5, 'active': True, ...}
                # **model_data unpacks it into keyword arguments
                # This is equivalent to: EnsembleMember(N_members=5, active=True, ...)
                config.ensemble_config[model_name] = EnsembleMember(**model_data)
        
        # === LOAD VARIABLE STATUS (variable_status.yaml) ===
        with open(variable_path, 'r') as f:
            variable_data = yaml.safe_load(f)
            
            # variable_data looks like: {'models': {'tas_annual_max': {'AWI-CM-1-1-MR': {...}, ...}, ...}}
            # Loop through each variable (tas_annual_max, tas_annual_mean, etc.)
            for var_name, models in variable_data['models'].items():
                config.variable_status[var_name] = {}
                
                # Loop through each model for this variable
                for model_name, model_data in models.items():
                    # Create a ModelVariable object and store it
                    config.variable_status[var_name][model_name] = ModelVariable(**model_data)
        
        return config
    
    def iter_active_models(self, variable: str) -> Iterator[ModelForAnalysis]:
        """
        THE MAIN METHOD FOR YOUR USE CASE!
        
        Iterate through models that are active for the given variable.
        This checks BOTH yaml files to make sure the model is:
        1. Active in models.yaml (ensemble_config)
        2. Active for this variable in variable_status.yaml
        
        Usage:
            for model in config.iter_active_models('tas_annual_max'):
                print(model.name)           # 'AWI-CM-1-1-MR'
                print(model.get_primary())  # 'AWI-CM-1-1-MR_r1i1p1f1'
        
        Args:
            variable: The variable name (e.g., 'tas_annual_max')
            
        Yields:
            ModelForAnalysis objects - one for each active model
        """
        # First check if the variable exists
        if variable not in self.variable_status:
            raise ValueError(f"Variable '{variable}' not found in configuration")
        
        # Loop through all models that have data for this variable
        for model_name, var_status in self.variable_status[variable].items():
            # Check 1: Is this model active for this variable?
            if not var_status.active:
                continue  # Skip to next model
            
            # Check 2: Does this model exist in ensemble config?
            if model_name not in self.ensemble_config:
                continue  # Skip to next model
            
            ensemble = self.ensemble_config[model_name]
            
            # Check 3: Is the model also active in ensemble config?
            if not ensemble.active:
                continue  # Skip to next model
            
            # All checks passed! Package everything up and yield it
            # 'yield' is like 'return' but for iterators - returns one at a time
            yield ModelForAnalysis(
                name=model_name,
                primary_member=ensemble.primary_member,
                all_members=ensemble.ensemble_members,
                valid_years=var_status.valid_years,
                failure_mode=var_status.failure_mode
            )
    
    def get_active_models_list(self, variable: str) -> List[ModelForAnalysis]:
        """
        Get a list of all active models for a variable.
        
        This is just a convenience wrapper around iter_active_models.
        Use this if you want a list instead of an iterator.
        
        Usage:
            models = config.get_active_models_list('tas_annual_max')
            print(f"Found {len(models)} active models")
        """
        return list(self.iter_active_models(variable))
    
    def is_model_active(self, model_name: str, variable: str) -> bool:
        """
        Check if a specific model is active for a variable.
        
        Quick yes/no check without getting all the data.
        
        Usage:
            if config.is_model_active('AWI-CM-1-1-MR', 'tas_annual_max'):
                print("It's active!")
        """
        # Check if variable exists
        if variable not in self.variable_status:
            return False
        # Check if model exists for this variable
        if model_name not in self.variable_status[variable]:
            return False
        # Check if model exists in ensemble config
        if model_name not in self.ensemble_config:
            return False
        
        # Return True only if active in both places
        return (self.variable_status[variable][model_name].active and 
                self.ensemble_config[model_name].active)
    
    def get_model_for_analysis(self, model_name: str, variable: str) -> Optional[ModelForAnalysis]:
        """
        Get a specific model's info for analysis if it's active.
        
        Usage:
            model = config.get_model_for_analysis('AWI-CM-1-1-MR', 'tas_annual_max')
            if model:
                print(f"Use member: {model.get_primary()}")
            else:
                print("Model not active")
        
        Returns:
            ModelForAnalysis object if active, None if not active
        """
        # First check if active
        if not self.is_model_active(model_name, variable):
            return None
        
        # Get the data from both configs
        ensemble = self.ensemble_config[model_name]
        var_status = self.variable_status[variable][model_name]
        
        # Package it up
        return ModelForAnalysis(
            name=model_name,
            primary_member=ensemble.primary_member,
            all_members=ensemble.ensemble_members,
            valid_years=var_status.valid_years,
            failure_mode=var_status.failure_mode
        )


# ==============================================================================
# EXAMPLE USAGE - Shows how to use the code above
# ==============================================================================
if __name__ == "__main__":
    # === STEP 1: Load both yaml files ===
    config = ClimateModelConfig.from_yaml_files(
        'config/meta.yaml',
        'config/qc.yaml'
    )
    
    print(config)
    # Now 'config' contains all the data from both files
    
    variable = 'tas_annual_max'
    
    # === MAIN USE CASE 1: Loop through all active models ===
    print(f"=== Analyzing {variable} with PRIMARY members only ===\n")
    
    # This loop goes through each active model one at a time
    for model in config.iter_active_models(variable):
        # 'model' is a ModelForAnalysis object with everything you need
        print(f"Model: {model.name}")
        print(f"  Primary member: {model.get_primary()}")
        print(f"  Valid years: {model.valid_years[0]} to {model.valid_years[-1]}")
        print(f"  # of years: {len(model.valid_years)}")
        
        # YOUR ANALYSIS CODE WOULD GO HERE
        # Example: load_data(model.get_primary(), model.valid_years)
        
        print()
    
    print("\n" + "="*60 + "\n")
    
    # === MAIN USE CASE 2: Loop through all active models using ALL members ===
    print(f"=== Analyzing {variable} with ALL ensemble members ===\n")
    
    for model in config.iter_active_models(variable):
        print(f"Model: {model.name}")
        print(f"  All members ({len(model.all_members)}): {model.all_members[:3]}...")
        
        # Loop through each ensemble member
        for member in model.get_all_members():
            # YOUR ANALYSIS CODE HERE for each member
            # Example: load_data(member, model.valid_years)
            pass
        
        print()
    
    print("\n" + "="*60 + "\n")
    
    # === ALTERNATE: Check a specific model ===
    print("=== Checking specific model: ACCESS-CM2 ===\n")
    
    model = config.get_model_for_analysis('ACCESS-CM2', variable)
    if model:
        # model is a ModelForAnalysis object
        print(f"ACCESS-CM2 is active for {variable}")
        print(f"Primary: {model.get_primary()}")
    else:
        # model is None
        print(f"ACCESS-CM2 is NOT active for {variable}")
    
    print("\n" + "="*60 + "\n")
    
    # === Get count of active models ===
    active_count = len(config.get_active_models_list(variable))
    print(f"Total active models for {variable}: {active_count}")