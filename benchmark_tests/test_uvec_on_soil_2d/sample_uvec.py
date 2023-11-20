import json

def uvec_test(json_string):
    uvec_data = json.loads(json_string)

    load_wheel_1 = uvec_data["parameters"]["load_wheel_1"]
    load_wheel_2 = uvec_data["parameters"]["load_wheel_2"]

    uvec_data['loads'] = {1: [0, load_wheel_1, 0], 2: [0, load_wheel_2, 0]}
    return json.dumps(uvec_data)
    