import os

def patient_paths(root):
    patients = os.listdir(root)
    patients = [os.path.join(root, name) for name in patients if 'pat' in name.lower()]
    return patients
