import json


def uvec_test(json_string: str) -> str:
    """
    Args:
        - json_string (str): json string containing the uvec data

    Returns:
        - str: json string containing the load data

    """

    # Get the uvec data
    uvec_data = json.loads(json_string)

    # Get the uvec parameters
    load_wheel_1 = uvec_data["parameters"]["load_wheel_1"]
    load_wheel_2 = uvec_data["parameters"]["load_wheel_2"]

    # Set the load data
    uvec_data['loads'] = {1: [0, load_wheel_1, 0], 2: [0, load_wheel_2, 0]}
    return json.dumps(uvec_data)
