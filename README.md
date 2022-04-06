# Antichromatizer

**WARNING**: No guarantees are made about the safety or stability of this product. Photosensitive individuals should exercise *EXTREME CAUTION* when viewing .gif files generated by this program, particularly with low-valued parameters.

Otherwise, a silly little program. Generates various antichromatic images from an input image, generates animations, and allows for directed blending. No command line arguments, there is a minimal UI once run.

You may utilize the .yml file to automatically set up the required Python environment with Anaconda, or you can manually install the required packages. You may want to update conda before creating the environment:

	conda update -n base conda
	conda env create -f environment.yml

Once the environment has been successfully created, activate it with:

	conda activate Antichromatizer

# Sample Images
We avoid embedding the animated items below, due to their strobe-like affect. They are still available to view [here](https://github.com/j-henshaw/Antichromatizer/tree/main/Examples).

![J - Original](https://github.com/j-henshaw/Antichromatizer/blob/main/Examples/j.jpg?raw=true)
![J - Threshold - Outliers - Inverse](https://github.com/j-henshaw/Antichromatizer/blob/main/Examples/j-THRESH_xOUTLIERS_inv.png)
![J - Threshold - Mode - Inverse](https://github.com/j-henshaw/Antichromatizer/blob/main/Examples/j-THRESH_xMODE_inv.png)
![J - Conglomeration](https://github.com/j-henshaw/Antichromatizer/blob/main/Examples/j_%5B0.8121%20-0.5576%200.0000%200.7515%200.0000%200.0000%200.2606%200.9697%20-0.4485%200.5152%200.6303%20-0.1091%200.0000%200.0000%200.0000%20-0.4061%5D_CONGLOM.png)
![A2 - Threshold - Outliers](https://github.com/j-henshaw/Antichromatizer/blob/main/Examples/A2-THRESH_OUTLIERS.png)
![A1 - Conglomeration](https://github.com/j-henshaw/Antichromatizer/blob/main/Examples/A1_%5B-0.4687%20-0.9325%20-0.9292%200.9686%200.0501%200.8271%20-0.6012%200.3860%20-0.8852%200.1659%200.2994%200.7968%20-0.5599%20-0.2550%20-0.1170%20-0.5059%5D_CONGLOM.png)
