"""Dataclass structure for CMIP6 climate model ensemble.

The point of this is to create a `dataclass` object that contains information about
what CMIP6 models have data for a given variable, and for how many years. The `dataclass`
ingests two .yaml files created after running `meta_qc_cmip6.py`.

Adam Michael Bauer
UChicago
1.26.2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Iterator
import yaml, warnings


@dataclass
class CMIP6EnsembleMember:
    """Climate model class based on metadata.
    """
    N_members: str  # number of members
    ensemble_members: list[str]  # ensemble members
    primary_member: str  # primary member, usually r1p1i1f1

    # some helper methods for a given ensemble member
    def get_primary_member(self) -> str:
        return self.primary_member
    
    def get_all_members(self) -> list[str]:
        return self.ensemble_members


@dataclass
class ModelVariable:
    """A class for model x variable information.

    For example, this class gives the info a model's data on a given variable.
    """
    active: bool  # is the model active for this variable?
    failure_mode: str  # if not, what's the failure mode?
    invalid_years: list[str]  # invalid years where we don't have data
    valid_years: list[str]  # valid years where we do have data


@dataclass
class ModelStagedForAnalysis:
    """Staged model for analysis. This is a class that aggreagates the useful information
    from CMIP6EnsembleMember and ModelVariable.
    """

    name: str  # model name
    valid_years: list[str]  # valid years to use in analysis
    all_members: list[str]  # all ensemble members
    primary_member: str  # primary ensemble member


@dataclass
class CMIP6EnsembleConfig:
    """Main datacalss that stores information about all of my climate model ensemble
    members.
    """

    # structure: (model_name, model object)
    ensemble_config: Dict[str, CMIP6EnsembleMember] = field(default_factory=dict)
    
    # structure: (variable_name, model_name, model x variable object)
    variable_config: Dict[str,
                          Dict[str, ModelVariable]] = field(default_factory=dict)

    
    @classmethod
    def from_yaml(cls, ensemble_config_path: str, variable_config_path: str) -> 'CMIP6EnsembleConfig':
        """Create a CMIP6EnsembleConfig from a pair of .yaml files.

        Parameters
        ----------
        ensemble_config_path: str
            the path to the ensemble config file, usually "config/meta.yaml"

        variable_config_path: str
            the path to the variable config file, usually "config/qc.yaml"

        Returns
        -------
        cls: CMIP6EnsembleConfig
            the method creates the CMIP6EnsembleConfig from the two .yaml files
        """

        config = cls()

        # load in data from yaml
        with open(ensemble_config_path, 'r') as f:
            ensemble_data = yaml.safe_load(f)

            # loop through and fill out the ensemble_config dict
            for model_name, model_info in ensemble_data['models'].items():
                config.ensemble_config[model_name] = CMIP6EnsembleMember(**model_info)

        # load in variable information
        with open(variable_config_path, 'r') as f:
            variable_config = yaml.safe_load(f)

            # loop through variables and each model for each variable
            # loop for each variable
            for variable_name, models in variable_config['models'].items():
                # empty dictionary for individual model information
                config.variable_config[variable_name] = {}

                # loop through each model and its info to populate information for current
                # variable
                for model_name, model_info in models.items():
                    config.variable_config[variable_name][model_name] = ModelVariable(**model_info)

        return config
    
    def iter_active_models(self, variable: str, verbose: bool = False) -> Iterator[ModelStagedForAnalysis]:
        """Main class method: iterate through active models for a given variable.

        Parameters
        ----------
        variable: str
            the variable we want to loop through

        verbose: bool
            whether or not one wants to raise warnings or not when function is called

        Yields
        ------
        ModelStagedForAnalysis: class
            model class with relevant information for analysis
        """

        # if variable is not in the current .yaml config, raise error
        if variable not in self.variable_config:
            raise ValueError("Desired variable not a data variable in CMIP6 ensemble data provided.")
        
        # loop through models and ModelVariable classes
        for model_name, modvar in self.variable_config[variable].items():
            # if model is not active, raise warning / skip
            if not modvar.active:
                if verbose:
                    warnings.warn("Note: for model {}, the variable {} is not active. Skipping.".format(model_name, variable)) 
                else:
                    continue

            # if model doesn't exist in the current config, raise warning / skip
            if model_name not in self.ensemble_config:
                if verbose:
                    warnings.warn("Model {} does not exist in current config. Skipping.".format(model_name))
                else:
                    continue
                
            # with the checks passed, we can yield an ensemble member class
            ens_member = self.ensemble_config[model_name]
        
            yield ModelStagedForAnalysis(
                name=model_name,
                valid_years=modvar.valid_years,
                all_members=ens_member.get_all_members(),
                primary_member=ens_member.primary_member
            )

    def get_active_models(self, variable: str) -> List[ModelVariable]:
        """Helper function that returns all active models for a given variable

        Parameters
        ----------
        variable: str
            variable for analysis

        Returns
        -------
        list of active models for the passed variable
        """

        return list(self.iter_active_models(variable))
    
    def is_model_active(self, model: str, variable: str) -> bool:
        """Boolean check function for if a model is active for a given variable.

        Parameters
        ----------
        model: str
            the CMIP model in question

        variable: str
            the variable to be analyzed

        Returns
        -------
        is_active: bool
            whether or not the model is active for the variable
        """

        # check if variable is in the config, if not, pass false
        if variable not in self.variable_config:
            warnings.warn("Variable is not in current config.")
            return False
        
        # check if model is in the config, if not, pass false
        if model not in self.variable_config[variable]:
            warnings.warn(f"Model is not registered in current config for variable {variable}. Returning False.")
            return False
        
        # else, return active attribute from ModelVariable class
        return self.variable_config[variable][model].active
    
    def get_model_for_analysis(self, model: str, variable: str) -> ModelStagedForAnalysis:
        """Fetch a given model and variable for analysis.

        Parameters
        ----------
        model: str
            the cmip model you want

        variable: str
            the variable you're interested in

        Returns
        -------
        ModelVariable: class
            the model x variable config class
        """

        # if the model isn't active, raise a ValueError
        if not self.is_model_active(model, variable):
            raise ValueError("Model {} is not active for variable {}".format(model, variable))
        
        # otherwise pull the ensemble member and the ModelVariable and make the 
        # ModelStagedForAnalysis class
        ens_member = self.ensemble_config[model]
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
