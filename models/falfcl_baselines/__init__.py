"""FALFCL/FACL-adapted baseline models written for the ICLR26 baseline sweep.

Every file here is NEW. None of the pre-existing model files were edited or
overwritten -- where an existing FACL adaptation was wrong or not resumable, a
new file was added alongside it and the original left untouched.

  alphapre_falfcl_v2.py   AlphaPre: FALFCL on the base regression term ONLY;
                          phase / amplitude / A-Net terms stay native, and the
                          amp_weight decay schedule is restored. (The existing
                          models/alphapre_falfcl.py replaces the whole objective
                          with FALFCL, which drops those three terms.)
  convlstm_falfcl.py      ConvLSTM + FALFCL on post-sigmoid predictions.
  phydnet_falfcl.py       PhyDNet: FALFCL on the decoder forecast term ONLY;
                          encoder reconstruction and the K2M moment constraint
                          stay MSE (FALFCL on a 7x7 moment matrix is meaningless
                          and not even 5D).
  earthfarseer_falfcl.py  EarthFarseer + FALFCL, with 20.7M dead parameters
                          removed (enc, skip_conneciton, Mlp.fc2 -- all verified
                          gradient-free; removal is output-identical).
  trajGRU_falfcl_v2.py    TrajGRU + FALFCL with the curriculum step persisted.
  earth_former_falfcl_v2.py  EarthFormer + FALFCL with the curriculum step persisted.
  simvp_falfcl_nt6.py     SimVP+FALFCL pinned to N_T=6, the value the completed
                          SimVP checkpoints were trained with (the shared file
                          now declares N_T=4 and cannot load them).
"""
