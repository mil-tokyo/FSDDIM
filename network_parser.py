import yaml


class parse(object):
    """
    This class reads yaml parameter file and allows dictionary like access to the members.
    """

    def __init__(self, path):
        with open(path, 'r') as file:
            params = yaml.safe_load(file)

        if 'variables' not in params:
            self.parameters = params
        else:
            self.parameters = self.parse_variables(params,
                                                   params.pop('variables'))

    def parse_variables(self, params: dict, variables: dict) -> dict:
        for key, val in params.items():
            if isinstance(val, dict):
                params[key] = self.parse_variables(val, variables)
            elif isinstance(val, str) and val.startswith('$'):
                params[key] = variables[val.lstrip('$')]
            else:
                params[key] = val

        return params

    # Allow dictionary like access
    def __getitem__(self, key):
        return self.parameters[key]

    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)
