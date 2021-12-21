import numpy as np
import h5py

Nfiles = 100
Npart = int(1e4)
boxl = 50.


for i in range(Nfiles):
    print('Processing file ',i)
    with h5py.File('snap_%03d.hdf5'%i,'w') as f:
        f.create_group('PartType1')

        ## Random uniform positions
        coods = np.random.rand(Npart,3) * boxl
        print('min, max coords = ',np.min(coods),np.max(coods))

        ## Random gaussian velocities
        vels = np.random.normal(loc=0.,scale=1.,size=(Npart,3))

        f.create_dataset('PartType1/Coordinates',data=coods)
        f.create_dataset('PartType1/Velocity',data=vels)

        

