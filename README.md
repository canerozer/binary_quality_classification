# Binary MRI Quality Classification
A repository for classifying low and high-quality MRI images.


# Usage
To show the slices of the 4D images, the user should first create a YAML config file that some examples are present in configs folder. Inside the YAML config file, several attributes are needed to be filled.

src: Represents the original folder containing the nii.gz files.
format: File format for the NifTI images.
process_ones_with: Suffix of the files that will be shown.
show_image: Shows the image after reading the image volume
show_kspace: Shows the k-space representation after taking the Fourier Transform of the image volume.
