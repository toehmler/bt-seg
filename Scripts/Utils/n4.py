from nipype.interfaces.ants import N4BiasFieldCorrection
import os
import ast

import SimpleITK as sitk

def bias_field_correction(path, n_dims, n_iters, out_path):
    """
    """
    n4 = n4biasfieldcorrection(output_image=out_path)
    n4.inputs.dimension = n_dims
    n4.input_image = path
    n4.inputs.n_iterations = ast.literal_eval(n_iters)
    print("running n4 bias correction for " + path);
    n4.run()
    print("finished")

