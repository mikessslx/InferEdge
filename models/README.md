This directory will store TorchScript models converted by the suite that will be transferred 
to target devices. You may also insert your own models in TorchScript format here, in addition to
what the suite can generate. Note there is a nested models directory to avoid storing the README 
together with the models, since they will be transferred to the target device, and every file in 
that nested directory will be considered as a model for the experiments' purposes.