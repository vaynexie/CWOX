# Code to produce a base heatmap using explanation method named algo.
# We provide algo : grad_cam_cwox (for cluster and class), excitationbackprop_cwox (for cluster and class), rise_cwox (for cluster), rise (for class), lime_cwox (for cluster), lime (for class).
# Users are also welcome to create their own algo following similar code style. 
def IOX(algo):
    def base_explainer(image,targets):
        return algo(image,targets)
    return base_explainer
