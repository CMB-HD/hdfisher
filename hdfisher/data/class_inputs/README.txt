Inputs that CLASS needs by absolute path.

    PRIMAT21_class_format.dat
        The PRIMAT 2021 BBN table, converted to the three-column CLASS
        format (ombh2, DeltaN, Yp) from CAMB's
        `PRIMAT_Yp_DH_ErrorMC_2021.dat`, so that CAMB and CLASS use the
        same helium abundance.

        Referenced as `sBBN file:` by
        ../fiducial_params/class_fiducial_params.yaml. CLASS requires an
        absolute path, so that file carries a `/path/to/hdfisher/...`
        placeholder that must be edited to point here.
