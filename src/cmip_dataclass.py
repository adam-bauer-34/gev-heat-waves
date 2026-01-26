from dataclasses import dataclass, field
from typing import List, Dict, Iterator
import yaml, warnings


@dataclass
class CMIP6EnsembleMember:
    """Climate model class based on metadata.
    """
    N_members: str
    ensemble_members: list[str]
    primary_member: str

    def get_primary_member(self) -> str:
        return self.primary_member
    
    def get_all_members(self) -> list[str]:
        return self.ensemble_members


@dataclass
class ModelVariable:
    """A class for model x variable information.

    For example, this class gives the info a model's data on a given variable.
    """
    active: bool
    failure_mode: str
    invalid_years: list[str]
    valid_years: list[str]


@dataclass
class ModelStagedForAnalysis:
    name: str
    valid_years: list[str]
    all_members: list[str]
    primary_member: str


@dataclass
class CMIP6EnsembleConfig:
    # structure: (model_name, model object)
    cmip6_ensemble_config: Dict[str, CMIP6EnsembleMember] = field(default_factory=dict)
    
    # structure: (variable_name, model_name, model x variable object)
    variable_config: Dict[str,
                          Dict[str, ModelVariable]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, ensemble_config_path: str, variable_config_path: str) -> 'CMIP6EnsembleConfig':
        config = cls()

        # load in data from yaml
        with open(ensemble_config_path, 'r') as f:
            ensemble_data = yaml.safe_load(f)

            # loop through and fill out the cmip6_ensemble_config dict
            for model_name, model_info in ensemble_data['models'].items():
                config.cmip6_ensemble_config[model_name] = CMIP6EnsembleMember(**model_info)

        with open(variable_config_path, 'r') as f:
            variable_config = yaml.safe_load(f)

            for variable_name, models in variable_config['models'].items():
                config.variable_config[variable_name] = {}

                for model_name, model_info in models.items():
                    config.variable_config[variable_name][model_name] = ModelVariable(**model_info)

        return config
    
    def iter_active_models(self, variable: str, verbose: bool = False) -> Iterator[ModelStagedForAnalysis]:
        if variable not in self.variable_config:
            raise ValueError("Desired variable not a data variable in CMIP6 ensemble data provided.")
        
        for model_name, modvar in self.variable_config[variable].items():
            if not modvar.active:
                if verbose:
                    warnings.warn("Note: for model {}, the variable {} is not active. Skipping.".format(model_name, variable)) 
                else:
                    continue

            if model_name not in self.cmip6_ensemble_config:
                if verbose:
                    warnings.warn("Model {} does not exist in current config. Skipping.".format(model_name))
                else:
                    continue
                
            ens_member = self.cmip6_ensemble_config[model_name]
        
            yield ModelStagedForAnalysis(
                name=model_name,
                valid_years=modvar.valid_years,
                all_members=ens_member.get_all_members(),
                primary_member=ens_member.primary_member
            )

    def get_active_models(self, variable: str) -> List[ModelVariable]:
        return list(self.iter_active_models(variable))
    
    def is_model_active(self, model: str, variable: str) -> bool:
        if variable not in self.variable_config:
            warnings.warn("Variable is not in current config.")
            return False
        
        if model not in self.variable_config[variable]:
            warnings.warn(f"Model is not registered in current config for variable {variable}. Returning False.")
            return False
        
        return self.variable_config[variable][model].active
    
    def get_model_for_analysis(self, model: str, variable: str) -> ModelVariable:
        if not self.is_model_active(model, variable):
            raise ValueError("Model {} is not active for variable {}".format(model, variable))
        
        ens_member = self.cmip6_ensemble_config[model]
        modvar = self.variable_config[variable][model]

        return ModelStagedForAnalysis(
            name=model,
            valid_years=modvar.valid_years,
            all_members=ens_member.get_all_members(),
            primary_member=ens_member.primary_member
        )
    
# ==============================================================================
# EXAMPLE USAGE - Shows how to use the code above
# ==============================================================================
if __name__ == "__main__":
    # === STEP 1: Load both yaml files ===
    config = CMIP6EnsembleConfig.from_yaml(
        'config/meta.yaml',
        'config/qc.yaml'
    )
    
    # Now 'config' contains all the data from both files

    variable = 'tas_annual_max'
    
    # === MAIN USE CASE 1: Loop through all active models ===
    print(f"=== Analyzing {variable} with PRIMARY members only ===\n")
    
    # This loop goes through each active model one at a time
    for model in config.iter_active_models(variable):
        # 'model' is a ModelForAnalysis object with everything you need
        print(f"Model: {model.name}")
        print(f"  Primary member: {model.primary_member}")
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
        for member in model.all_members:
            # YOUR ANALYSIS CODE HERE for each member
            # Example: load_data(member, model.valid_years)
            pass
        
        print()
    
    print("\n" + "="*60 + "\n")
    
    # === ALTERNATE: Check a specific model ===
    test_model = 'ACCESS-CM2'
    print("=== Checking specific model: {} ===\n".format(test_model))
    
    try:
        model = config.get_model_for_analysis('ACCESS-CM2', variable)
        
        # model is a ModelForAnalysis object
        print("{} is active for {}".format(test_model, variable))
        print(f"Primary: {model.primary_member}")

    except ValueError:
        # model is None
        print("{} is NOT active for {}".format(test_model, variable))

    
    print("\n" + "="*60 + "\n")
    
    # === Get count of active models ===
    active_count = len(config.get_active_models(variable))
    print(f"Total active models for {variable}: {active_count}")