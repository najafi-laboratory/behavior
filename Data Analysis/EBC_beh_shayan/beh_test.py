import h5py 
import scipy.io as sio
import numpy as np

def mat_to_dict(mat_file):
    """Loads a .mat file and converts it into a nested Python dictionary."""
    
    def check_keys(d):
        """Recursively converts MATLAB structs to Python dictionaries."""
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = tolist(d[key])
        return d

    def todict(matobj):
        """Converts a MATLAB struct object to a Python dictionary."""
        d = {}
        for field in getattr(matobj, '_fieldnames', []):  # Ensure _fieldnames exists
            elem = getattr(matobj, field, None)
            if isinstance(elem, sio.matlab.mat_struct):
                d[field] = todict(elem)
            elif isinstance(elem, np.ndarray):
                d[field] = tolist(elem)
            else:
                d[field] = elem
        return d

    def tolist(ndarray):
        elem_list = []
        if ndarray.ndim == 0:  # Handle 0-d arrays
            return ndarray.item()  # Convert to Python scalar
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
        # return [tolist(elem) if isinstance(elem, np.ndarray) else elem for elem in ndarray]

    mat_data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    return check_keys(mat_data)

# Usage
bpod_file = './data/beh/E5LG/raw/E5LG_EBC_V_3_12_20241024_160443.mat'
# bpod_file = './data/beh/E4L7/raw/E4L7_EBC_V_3_11_20240901_173924.mat'
raw = mat_to_dict(bpod_file)

# print(np.array(raw['SessionData']['RawEvents']['Trial'][0]))

for item in raw['SessionData']:
    print(item)
# breakpoint()

print(raw['SessionData']['ExperimenterInitials'])

print('the problem: \n')
print(raw['SessionData']['RawEvents']['Trial'][59]['Events'])

print('the okay \n')
print(raw['SessionData']['RawEvents']['Trial'][58]['Events'])

# # for item in raw.get('SessionData', []):  # Use .get() to avoid KeyError
# #     print(item)

# trials = h5py.File(f"./data/beh/E5LG/processed/E5LG_EBC_V_3_17_20250401_222431.h5")["trial_id"]

# #this does not exist!
# print(trials['114'])
